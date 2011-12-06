#ifndef GPU_VCD_SF_MP_K1_V2_CUH
#define GPU_VCD_SF_MP_K1_V2_CUH

#include "vcd_common.h"
#include "gpu_vcd_common.cuh"

// stream G channel calculation on R pixel, G channel calculation on B pixel, together with existing color into output
void stream_g_calculation(
	float* rg,
	float* bg,
	size_t outptc,
	float* tempvarh, float* tempvarv, float* tempvarbh, float* tempvarbv, size_t tempvarptc,
	const float* tempgh, const float* tempgv, const float* tempgd, size_t tempgptc,
	const float* rr, const float* bb, size_t inputptc, 
	const int* e, size_t eptc,
	int filter, vcd_params params,
	cudaDeviceProp prop, vcd_sf_mp_kernel_times* times);

// master function for calculating variance in x < 0 and y < 0 position.
// this function runs in diagonal order
void interpolate_g_dg_neg2(float* out, size_t outptc,
	float* tempvarh, float* tempvarv, float* tempvarbh, float* tempvarbv, size_t tempvarptc,
	const float* tempgh, const float* tempgv, const float* tempgd, size_t tempgptc,
	const float* input, size_t inputptc,
	const int* e, size_t eptc,
	int channel, int filter, vcd_params params, cudaDeviceProp prop, cudaStream_t stream, float* time);

__global__ void interpolate_g_dg_neg_k2(
	float* out, size_t outptc,
	float* tempvarh, float* tempvarv, float* tempvarbh, float* tempvarbv, size_t tempvarptc,
	const float* tempgh, const float* tempgv, const float* tempgd, size_t tempgptc,
	const float* input, size_t inputptc,
	const int* e, size_t eptc,
	int channel, int direction, int idx, int filter, vcd_params params, smeminfo_interpolate_g_neg smeminfo,
	int major, int minor);

#endif