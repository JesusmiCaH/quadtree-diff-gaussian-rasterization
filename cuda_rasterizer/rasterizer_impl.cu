/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"


// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}


__device__ bool BoxInsideGaussian(
	uint2 rect_min, uint2 rect_max,
	float2 points_xy,
	float3 cov2D_inv,
	float confidence,
)
{
	// Transform the rectangle corners relative to the Gaussian center
	int2 c_rect_min = { rect_min.x - points_xy.x, rect_min.y - points_xy.y };
	int2 c_rect_max = { rect_max.x - points_xy.x, rect_max.y - points_xy.y };

	// Check each corner of the rectangle
	float corners[4][2] = {
		{ c_rect_min.x, c_rect_min.y },
		{ c_rect_max.x, c_rect_min.y },
		{ c_rect_min.x, c_rect_max.y },
		{ c_rect_max.x, c_rect_max.y }
	};

	uint n_inside = 0;

	for (int i = 0; i < 4; i++) {
		float x = corners[i][0];
		float y = corners[i][1];


		// Evaluate the ellipse equation: cov2D.x * x^2 + 2 * cov2D.y * x * y + cov2D.z * y^2 <= 1
		float ellipse_value = cov2D_inv.x * x * x + 2.0f * cov2D_inv.y * x * y + cov2D_inv.z * y * y;
		// printf("ellipse value is %f\n", ellipse_value);
		if (ellipse_value < confidence) {
			n_inside++;
			if(n_inside>=2) return true;
		}
	}
	// All corners are inside the ellipse
	return false;
}

__device__ bool GaussianInsideBox(
	uint2 rect_min, uint2 rect_max,
	float2 points_xy)
{
	return (points_xy.x >= rect_min.x && points_xy.x < rect_max.x &&
		points_xy.y >= rect_min.y && points_xy.y < rect_max.y);
}

__device__ bool GaussianIntersectBox(
	uint2 rect_min, uint2 rect_max,
	float2 points_xy,
	float3 cov2D_inv,
	float tolerant,
)
{
	float2 c_rect_min = { rect_min.x, rect_min.y };
	float2 c_rect_max = { rect_max.x, rect_max.y };

	c_rect_max.x -= points_xy.x;
	c_rect_max.y -= points_xy.y;
	c_rect_min.x -= points_xy.x;
	c_rect_min.y -= points_xy.y;

	
	
	float delta_n = c_rect_min.y * c_rect_min.y * cov2D_inv.y * cov2D_inv.y - cov2D_inv.x * ( cov2D_inv.z * c_rect_min.y * c_rect_min.y -1);
	if (delta_n >= 0){
		float x_min = (- cov2D_inv.y * c_rect_min.y - sqrt(delta_n)) / cov2D_inv.x;
		float x_max = (- cov2D_inv.y * c_rect_min.y + sqrt(delta_n)) / cov2D_inv.x;
		if (x_min < c_rect_max.x+tolerant && x_max > c_rect_min.x-tolerant)
			return true;
	}

	float delta_s = c_rect_max.y * c_rect_max.y * cov2D_inv.y * cov2D_inv.y - cov2D_inv.x * ( cov2D_inv.z * c_rect_max.y * c_rect_max.y -1);
	if (delta_s >= 0){
		float x_min = (- cov2D_inv.y * c_rect_max.y - sqrt(delta_s)) / cov2D_inv.x;
		float x_max = (- cov2D_inv.y * c_rect_max.y + sqrt(delta_s)) / cov2D_inv.x;
		if (x_min < c_rect_max.x+tolerant && x_max > c_rect_min.x-tolerant)
			return true;
	}

	float delta_w = c_rect_min.x * c_rect_min.x * cov2D_inv.y * cov2D_inv.y - cov2D_inv.z * ( cov2D_inv.x * c_rect_min.x * c_rect_min.x -1);
	if (delta_w >= 0){
		float y_min = (- cov2D_inv.y * c_rect_min.x - sqrt(delta_w)) / cov2D_inv.z;
		float y_max = (- cov2D_inv.y * c_rect_min.x + sqrt(delta_w)) / cov2D_inv.z;
		if (y_min < c_rect_max.y+tolerant && y_max > c_rect_min.y-tolerant)
			return true;
	}

	float delta_e = c_rect_max.x * c_rect_max.x * cov2D_inv.y * cov2D_inv.y - cov2D_inv.z * ( cov2D_inv.x * c_rect_max.x * c_rect_max.x -1);
	if (delta_e >= 0){
		float y_min = (- cov2D_inv.y * c_rect_max.x - sqrt(delta_e)) / cov2D_inv.z;
		float y_max = (- cov2D_inv.y * c_rect_max.x + sqrt(delta_e)) / cov2D_inv.z;
		if (y_min < c_rect_max.y+tolerant && y_max > c_rect_min.y-tolerant)
			return true;
	}
	
	return false;
}

#define MAX_STACK 64

__device__ void kvpairByQuadtree(
	uint2 rect_min, uint2 rect_max,
	dim3 grid,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	uint32_t offset,
	uint32_t idx,
	float2 points_xy,
	const float* depths,
	float3 cov2D,
	float tolerant,
	float confidence
)
{
	// 模拟栈
	uint2 stack_min[MAX_STACK];
	uint2 stack_max[MAX_STACK];
	int stack_ptr = 0;

	// 初始压入
	stack_min[stack_ptr] = rect_min;
	stack_max[stack_ptr] = rect_max;
	stack_ptr++;

	while (stack_ptr > 0)
	{
		stack_ptr--;
		uint2 cur_min = stack_min[stack_ptr];
		uint2 cur_max = stack_max[stack_ptr];

		uint2 real_rect_min = {cur_min.x * BLOCK_X, cur_min.y * BLOCK_Y};
		uint2 real_rect_max = {cur_max.x * BLOCK_X, cur_max.y * BLOCK_Y};
		// Compute the inverse of the 2D covariance matrix
		float det = cov2D.x * cov2D.z - cov2D.y * cov2D.y;
		float3 cov2D_inv = {
			cov2D.z / det,
			-cov2D.y / det,
			cov2D.x / det
		};

		if (
			GaussianInsideBox(real_rect_min, real_rect_max, points_xy) ||
			GaussianIntersectBox(real_rect_min, real_rect_max, points_xy, cov2D_inv, tolerant) ||
			BoxInsideGaussian(real_rect_min, real_rect_max, points_xy, cov2D_inv, confidence) ||
			false
		)
		{
			if (cur_max.x - cur_min.x == 0 || cur_max.y - cur_min.y == 0)
			{
				continue;
			}
			else if (cur_max.x - cur_min.x == 1 && cur_max.y - cur_min.y == 1)
			{
				uint64_t key = cur_min.y * grid.x + cur_min.x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[offset] = key;
				gaussian_values_unsorted[offset] = idx;
				offset++;
			}
			else if (stack_ptr + 4 <= MAX_STACK)
			{
				uint2 mid = {
					(cur_min.x + cur_max.x) / 2,
					(cur_min.y + cur_max.y) / 2};

				// 四个象限压入栈
				stack_min[stack_ptr] = cur_min;
				stack_max[stack_ptr] = mid;
				stack_ptr++;

				stack_min[stack_ptr] = {mid.x, cur_min.y};
				stack_max[stack_ptr] = {cur_max.x, mid.y};
				stack_ptr++;

				stack_min[stack_ptr] = {cur_min.x, mid.y};
				stack_max[stack_ptr] = {mid.x, cur_max.y};
				stack_ptr++;

				stack_min[stack_ptr] = mid;
				stack_max[stack_ptr] = cur_max;
				stack_ptr++;
			}
		}
	}
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	float3* cov2Ds,
	dim3 grid,
	float tolerant,
	float confidence)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		uint32_t off_tree = off;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// printf("idx %lld, offset %d \n", idx, off);
		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 

		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				// uint64_t key = y * grid.x + x;
				// printf("key is %llu, value is %d\n", key, y * grid.x + x);
				// key <<= 32;
				// key |= *((uint32_t*)&depths[idx]);
				// gaussian_keys_unsorted[off] = key;
				gaussian_keys_unsorted[off] = 0;
				// gaussian_values_unsorted[off] = idx;
				// printf("offset = %d, gaussiankeysunsorted = %lld \n", off, gaussian_keys_unsorted[off]);
				off++;	
			}

		}


		// Use quadtree to find the tile that the Gaussian intersects
		// and add the key/value pair to the list.
		kvpairByQuadtree(
			rect_min, rect_max,
			grid,
			gaussian_keys_unsorted,
			gaussian_values_unsorted,
			off_tree,
			idx,
			points_xy[idx],
			depths,
			cov2Ds[idx],
			tolerant,
			confidence,
		);

	}
}


// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();

	if (idx >= L){
		return;
	}

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	if (key==0){
		return;
	}
	uint32_t currtile = key >> 32;
	uint32_t prevtile = point_list_keys[idx - 1] >> 32;

	// printf("IDX %lld, istrue? %d \n", idx, currtile != prevtile);

	if (idx == 0 || point_list_keys[idx - 1] == 0){
		ranges[currtile].x = idx;
	}
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;

		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
	// printf("range = %d, %d\n", ranges[currtile].x, ranges[currtile].y);
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	float* depth,
	bool antialiasing,
	int* radii,
	bool debug,
	float tolerant,
	float confidence)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	float3* cov2Ds;
	CHECK_CUDA(cudaMalloc((void**)&cov2Ds, P * sizeof(float3)), debug);

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		cov2Ds,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered,
		antialiasing
	), debug)

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	printf("numrenderd %d\n", num_rendered);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		cov2Ds,
		tile_grid,
		tolerant,

	);
	
	cudaDeviceSynchronize();

	CHECK_CUDA(cudaGetLastError(), debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)
	
	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// // Debugging: Check all the values inside binningState.point_list_keys
	// std::vector<uint64_t> h_point_list_keys(num_rendered);
	// cudaMemcpy(h_point_list_keys.data(), binningState.point_list_keys, num_rendered * sizeof(uint64_t), cudaMemcpyDeviceToHost);

	// for (int i = 0; i < num_rendered; ++i) {
	// 	printf("point_list_keys[%d] = %llu\n", i, h_point_list_keys[i]>>32);
	// }

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// Debugging: Check the imgState.ranges
	std::vector<uint2> h_ranges(tile_grid.x * tile_grid.y);
	cudaMemcpy(h_ranges.data(), imgState.ranges, tile_grid.x * tile_grid.y * sizeof(uint2), cudaMemcpyDeviceToHost);

	for (int i = 0; i < tile_grid.x * tile_grid.y; ++i) {
		printf("不应如是");
		printf("Tile %d: range start = %u, range end = %u\n", i, h_ranges[i].x, h_ranges[i].y);
	}

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		geomState.depths,
		depth), debug)

	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	const float* dL_invdepths,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dinvdepth,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	bool antialiasing,
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		geomState.depths,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		dL_invdepths,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor,
		dL_dinvdepth), debug);

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		opacities,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		dL_dinvdepth,
		dL_dopacity,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot,
		antialiasing), debug);
}
