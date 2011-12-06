#pragma once

#include "vcd_common.h"
#include "gpu_vcd_common.cuh"

//////////////////////////////////////////////////////////////////////////
// gpu_vcd_sf_mp_k2_v1.cuh
// gpu: file run on gpu
// vcd: file for VCD algorithm
// sf: single frame version
// mp: multi kernel version
// k1: kernel for calculating green channel
// v1: version 1
//////////////////////////////////////////////////////////////////////////

// stream G channel calculation on R pixel, G channel calculation on B pixel, together with existing color into output
void stream_g_calculation(
	float* outr,  
	float* outg, 
	float* outb, 
	size_t outptc,
	float* tempvarh, float* tempvarv, float* tempvarbh, float* tempvarbv, size_t tempvarptc,
	const float* tempgh, const float* tempgv, const float* tempgd, size_t tempgptc,
	const float* input, size_t inputptc, 
	const int* e, size_t eptc,
	int filter, vcd_params params,
	cudaDeviceProp prop, vcd_sf_mp_kernel_times* times);

// a master function for copying existing pixel from input matrix into separate channel matrix
// the technique is same, stage the data in shared memory first and then write to separate channel
void copy_to_channel(float* outr, float* outg, float* outb, size_t outptc,
	const float* input, size_t inputptc,
	int filter, vcd_params params, cudaStream_t stream);

// the worker kernel for copy_to_channel master function
__global__ void copy_to_channel_k(float* outr, float* outg,	float* outb, size_t outptc,
	const float* input, size_t inputptc,
	size_t smemptc,
	int filter, vcd_params params);

// master function for calculating variance in x < 0 and y < 0 position.
// this function runs in diagonal order
void interpolate_g_dg_neg(float* out, float* outg, size_t outptc,
	float* tempvarh, float* tempvarv, float* tempvarbh, float* tempvarbv, size_t tempvarptc,
	const float* tempgh, const float* tempgv, const float* tempgd, size_t tempgptc,
	const float* input, size_t inputptc,
	const int* e, size_t eptc,
	int channel, int filter, vcd_params params, cudaDeviceProp prop, cudaStream_t stream, float* time);

void interpolate_g_rest(const float* input, size_t inputptc,
	const int* e, size_t eptc);

__global__ void interpolate_g_dg_neg_k(
	float* out, float* outg, size_t outptc,
	float* tempvarh, float* tempvarv, float* tempvarbh, float* tempvarbv, size_t tempvarptc,
	const float* tempgh, const float* tempgv, const float* tempgd, size_t tempgptc,
	const float* input, size_t inputptc,
	const int* e, size_t eptc,
	int channel, int direction, int idx, int filter, vcd_params params, smeminfo_interpolate_g_neg smeminfo,
	int major, int minor);

// interpolate_g_dg_pos version 1: this version has strided loading from global memory
void interpolate_g_dg_pos(float* out, size_t outptc,
	float* tempvarh, float* tempvarv, float* tempvarbh, float* tempvarbv, size_t tempvarptc,
	const float* tempgh, const float* tempgv, const float* tempgd, size_t tempgptc,
	const float* input, size_t inputptc,
	const int* e, size_t eptc,
	int filter, vcd_params params, cudaDeviceProp prop, vcd_sf_mp_kernel_times* times);

__global__ void interpolate_g_dg_pos_k(
	float* out, size_t outptc,
	float* tempvarh, float* tempvarv, float* tempvarbh, float* tempvarbv, size_t tempvarptc,
	const float* tempgh, const float* tempgv, const float* tempgd, size_t tempgptc,
	const float* input, size_t inputptc,
	const int* e, size_t eptc,
	int filter, vcd_params params, smeminfo_interpolate_g_pos smeminfo, int major, int minor);

///////////////////////////////////////// VERSION 2: per-channel input ///////////////////////////////////////// 

// interpolate_g_dg_pos version 2: this version loads from local memory
void interpolate_g_dg_pos(float* out, size_t outptc,
	float* tempvarh, float* tempvarv, float* tempvarbh, float* tempvarbv, size_t tempvarptc,
	const float* tempgh, const float* tempgv, const float* tempgd, size_t tempgptc,
	const float* rr, const float* bb, size_t inputptc,
	const int* e, size_t eptc,
	int filter, vcd_params params, cudaDeviceProp prop, vcd_sf_mp_kernel_times* times);

__global__ void interpolate_g_dg_pos_k(
	float* out, size_t outptc,
	float* tempvarh, float* tempvarv, float* tempvarbh, float* tempvarbv, size_t tempvarptc,
	const float* tempgh, const float* tempgv, const float* tempgd, size_t tempgptc,
	const float* rr, const float* bb, size_t inputptc,
	const int* e, size_t eptc,
	int filter, vcd_params params, smeminfo_interpolate_g_pos smeminfo, int major, int minor);