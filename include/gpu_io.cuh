#ifndef GPU_CONVERT_INPUT_H
#define GPU_CONVERT_INPUT_H

#include "vcd_common.h"
#include "gpu_vcd_common.cuh"

// a kernel for converting unsigned short array into float array of raw image.
__global__ void convert_input_k (float* input_g, size_t input_g_pitch, 
	const unsigned short *librawInput, size_t librawInputPitch, 
	size_t smemPitch, int filter, vcd_params params);

// a kernel for testing conver_input kernel. Unoptimized.
__global__ void convert_input_test(unsigned int* output_R, 
	unsigned int* output_G, unsigned int* output_B, size_t output_pitch, 
	float* input_g, size_t input_g_pitch, vcd_params params);

__global__ void convert_input_test2(unsigned short* output, size_t output_pitch, const float* input_g, size_t input_g_pitch, vcd_params params);

// a kernel for copying output from separate R,G, and B array into one R, G, and B array
__global__ void convert_output_k(unsigned short* output, size_t output_pitch,
	const float* output_R, const float* output_G,
	const float* output_B, size_t output_sc_pitch, size_t smemPitch, vcd_params params);

// separate input to avoid strided access of global memory
__global__ void separate_input_k(float* rr, float* bb, float* grg, float* gbg, size_t chptc, 
								 const float* input, size_t inputptc, size_t inputptcs, int filter, vcd_params params);

// launcher for separate_input_k kernel
void separate_input(float* rr, float* bb, float* grg, float* gbg, size_t chptc,
					const float* input, size_t inputptc, int filter, vcd_params params, cudaDeviceProp prop, vcd_sf_mp_kernel_times* times);

// launcher for convert_input_k kernel
void convert_input(float* input_g, size_t input_g_pitch,
				   const unsigned short* librawINput, size_t librawInputPitch,
				   int filter, vcd_params params, cudaDeviceProp prop, vcd_sf_mp_kernel_times* times);

// launcher for convert_output_k kernel
void convert_output(unsigned short* output, size_t output_pitch,
	const float* output_R, const float* output_G,
	const float* output_B, size_t output_sc_pitch, vcd_params params, cudaDeviceProp prop, vcd_sf_mp_kernel_times* times);

// combine input kernel
__global__ void combine_channels_k(unsigned short* output, size_t outptc,
								 float *rr, float* rg, float* rb,
								 float *br, float* bg, float* bb,
								 float *grr, float* grg, float* grb,
								 float* gbr, float* gbg, float* gbb, size_t inchptc,
								 size_t stgptc,
								 int filter,
								 vcd_params params);

// launcher for combine_channels_k kernel
void combine_channels(unsigned short* output, size_t outptc,
					  float *rr, float* rg, float* rb,
								 float *br, float* bg, float* bb,
								 float *grr, float* grg, float* grb,
								 float* gbr, float* gbg, float* gbb, size_t inchptc, int filter, vcd_params params, cudaDeviceProp prop,
								 vcd_sf_mp_kernel_times* times);

#endif