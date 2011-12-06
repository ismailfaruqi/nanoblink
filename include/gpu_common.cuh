/****************************************************
* Accelerated VCD Algorithm
* File Name: gpu_vcd_common.cuh
* Contents:
* Constants for VCD algorithm implementation on GPU
* Author: Muhammad Ismail Faruqi
*****************************************************/
#ifndef GPU_COMMON_CUH
#define GPU_COMMON_CUH

#include <cuda_runtime.h>
#include <cuda.h>
#include <cutil.h>
#include "vcd_common.h"

#define COLOR_RED			0 // Constant for red cell
#define COLOR_GREEN_RED		1 // Constant for green cell in red row
#define COLOR_BLUE			2 // Constant for blue cell
#define COLOR_GREEN_BLUE	3 // Constant for green cell in bue row
#define COLOR_GREEN			1 // Constant for green cell

// transform 4-pixel tuple raw file used in LibRaw to 1-pixel raw tuple using GPU.
// 1 kernel handles 1 output pixel
template <class OutputType>
__global__ void libraw_image_to_flat_raw_kernel(OutputType* dst, unsigned short (*src)[4], int width, int height) 
{
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int x = blockDim.x * blockIdx.x + threadIdx.x;

	// read
}

// bdimx: block dimension, x
// bdimy: block dimension, y
template <class OutputType>
void libraw_image_to_flat_raw_gpu(OutputType* dst, unsigned short (*src)[4], int width, int height, int bdimx, int bdimy)
{
	dim3 dimBlock(bdimx, bdimy);
	dim3 dimGrid(width/(bdimx*1), height/(bdimy*1));
	libraw_image_to_flat_raw_kernel<OutputType><<<dimGrid, dimBlock>>>(dst, src, width, height);
}

// return the filter color of given row (y-coordinate) and col(x-coordinate)
inline unsigned short FC(int filter, int row, int col) 
{ 
	return (filter >> ((((row) << 1 & 14) + ((col) & 1)) << 1) & 3);
}

__device__ inline int fc(int filters, int x, int y)
{
	return (filters >> ((((y) << 1 & 14) + ((x) & 1)) << 1) & 3);
}

// get difference of a memory address from nearest aligned address in byte
__device__ inline int get_gmem_alignment_diff(size_t addr, int major, int minor)
{
	if (major == 2)
		return 0;//addr % 128; // 128 byte for CC 2.x
	else
		return addr % 64; // 64 byte for CC 1.x
}

// get maximum apron radius
__device__ inline int vcd_get_maxrad_k(vcd_params params)
{
	return max(params.temp_green_radius + params.window_radius, params.e_radius);
}


// get y correction given filter color target and source
__device__ inline int correct_y(vcd_params params, int tgtcolor, int thiscolor)
{
	if (thiscolor == COLOR_BLUE || COLOR_GREEN_BLUE)
	{
		if (tgtcolor == COLOR_RED || COLOR_GREEN_RED)
		{
			return params.gbtor;
		} else
			return 0;
	} else // thiscolor == COLOR_RED || COLOR_GREEN_RED
	{
		if (tgtcolor == COLOR_BLUE || COLOR_GREEN_BLUE)
		{
			return params.grtob;
		} else
			return 0;
	}
}

// get x correction given filter color target and source
__device__ inline int correct_x(vcd_params params, int tgtcolor, int thiscolor)
{
	if (thiscolor == COLOR_RED || COLOR_GREEN_BLUE)
	{
		if (tgtcolor == COLOR_BLUE || COLOR_GREEN_RED)
		{
			return params.gbtob;
		} else
			return 0;
	} else // thiscolor == COLOR_BLUE || COLOR_GREEN_RED
	{
		if (tgtcolor == COLOR_RED || COLOR_GREEN_BLUE)
		{
			return params.grtor;
		} else
			return 0;
	}
}

// helper function to get shared memory pitch given a width
size_t get_smem_pitch(size_t sz);



#endif