#ifndef GPU_VCD_SF_MP_K2_CUH
#define GPU_VCD_SF_MP_K2_CUH

#include "vcd_common.h"

// calculate R and B pixel at G pixels
__global__ void calculate_rest_k(float* grr, float grb, float* gbr, float *gbb, float* br, float* rb, 
									const float* grg, const float* gbg, const float* rg, const float* bg, const float* rr, const float* bb, size_t chptc, size_t chptcs, int filter, vcd_params params);

// launcher for calculate_rb_at_g_k kernel
void calculate_rest(float* grr, float* grb, float* gbr, float *gbb, float* br, float* rb,
					   const float* grg, const float* gbg, const float* rg, const float* bg, const float* rr, const float* bb, size_t chptc, 
					   int filter, vcd_params params, cudaDeviceProp prop, vcd_sf_mp_kernel_times *times);

#endif
