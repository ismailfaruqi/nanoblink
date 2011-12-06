#ifndef GPU_VCD_SF_MP_K0_H
#define GPU_VCD_SF_MP_K0_H

#include "vcd_common.h"

// kernel to calculate temporary green 
__global__ void calculate_temp_green_k(float* input_g, size_t input_g_pitch, 
	float* tempGreenV_g, float* tempGreenH_g, float* tempGreenD_g, 
	size_t tempGreen_g_pitch, size_t tempGreen_s_pitch, size_t staging_pitch, vcd_params params, int filter);

// launcher for temporary green kernel
void calculate_temp_green(float* input_g, size_t input_g_pitch, 
	float* tempGreenV_g, float* tempGreenH_g, float* tempGreenD_g, 
	size_t tempGreen_g_pitch, vcd_params params, int filter, cudaDeviceProp prop, vcd_sf_mp_kernel_times* times);

// kernel to calculate e
__global__ void calculate_e_k(float* e, size_t eptc, float* input, size_t inputptc, size_t smemptc, size_t stgptc, vcd_params params, int filter);

// kernel to test temporary green.
// copy green values from input array and temporary green array
__global__ void test_temp_green_k(float* outg, size_t outg_ptc, const float* tempg, size_t tempg_ptc, 
	const float* input, size_t input_ptc, vcd_params params, int filter);

// kernel to test e values
// give values to red channel that the pixel is an edge
__global__ void test_e_k(float* outr, size_t outrptc, int* e, size_t eptc, vcd_params params, int filter);

// function to calculate e
void calculate_e(int* e, size_t eptc, float* input, size_t inputptc, vcd_params params, int filter, cudaDeviceProp prop, vcd_sf_mp_kernel_times* times);

// function to test temporary green
void test_temp_green(float* outg, size_t outg_ptc, const float* tempg, size_t tempg_ptc, 
	const float* input, size_t input_ptc, vcd_params params, int filter, cudaDeviceProp prop);

// function to test e calculation result
void test_e(float* outr, size_t outrptc, int* e, size_t eptc, vcd_params params, int filter, cudaDeviceProp prop);

#endif