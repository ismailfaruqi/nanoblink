/****************************************************
* Accelerated VCD Algorithm
* File Name: gpu_vcd_common.cuh
* Contents:
* Constants for VCD algorithm implementation on GPU
* Author: Muhammad Ismail Faruqi
*****************************************************/
#ifndef GPU_VCD_COMMON_CUH
#define GPU_VCD_COMMON_CUH

#include <cuda_runtime.h>
#include <cuda.h>
#include <cutil.h>
#include "common.h"

#define EDGE		0x100
#define HORIZONTAL	0x001
#define VERTICAL	0x010
#define DIAGONAL	0x011
#define NON_EDGE	0x000

#define VERT_EDGE_PRE_CALC	1
#define HORZ_EDGE_PRE_CALC	2
#define TEX_PRE_CALC		3
#define FINISHED			4

#define DOWN_SWEEP	1
#define RIGHT_SWEEP 2
#define PARALLEL	3

typedef struct {
	int left_radius;
	int right_radius;
	int top_radius;
	int bottom_radius;
} transfer_params;

// channel definitions
#define RR 0
#define RG 1
#define RB 2
#define GRR 3
#define GRG 4
#define GRB 5
#define BR 6
#define BG 7
#define BB 8
#define GBR 9
#define GBG 10
#define GBB 11

// shared memory information for green interpolation kernel at negative directions.
typedef struct smeminfo_interpolate_g_neg {
	size_t inputptcs; // input pitch in shared memory. Load only corresponding channel.
	size_t outputgptcs; // output pitch in shared memory. Has apron on top and left side. Load and store only corresponding row.
	size_t tempgptcs; // temp g in shared memory.
	size_t tempglen; // temp g array length
	size_t varptcs; // positive direction variance element in shared memory. Has no apron
};

// shared memory information for green interpolation kernel at negative directions.
typedef struct smeminfo_interpolate_g_pos {
	size_t inputptcs; // input pitch in shared memory. Load only corresponding channel.
	size_t tempgptcs; // output pitch in shared memory. Has apron on top and left side. Load and store only corresponding row.
};

inline void smeminfo_interpolate_g_neg_init(smeminfo_interpolate_g_neg *info)
{
	info->inputptcs = 0;
	info->outputgptcs = 0;
}

inline void smeminfo_interpolate_g_pos_init(smeminfo_interpolate_g_pos *info)
{
	info->inputptcs = 0;
	info->tempgptcs = 0;
}

__device__ inline void vcd_sf_mp_determinte_g_k(float* outputg_s, size_t outputgptcs,
				const float* tempgh_s, const float* tempgv_s, const float* tempgd_s, size_t tempgptcs, size_t tempglen,
				float varh, float varv, float vard, int tdx, int tdy, int window_radius)
{
	int sx = tdx + window_radius / 2;
	int sy = tdy + window_radius / 2;

	int a = (varv < varh); // 1-a -> varh <= varv
	int b = (vard < varh); // 1-b -> varh <= vard
	int c = (varv < vard); // 1-c -> vard <= varv

	int i = 1 * (a & c) + 2 * (b & (1-c));

	outputg_s[sy * (outputgptcs / sizeof(float)) + sx] = tempgh_s[i * (tempglen / sizeof(float)) + tdy * (tempgptcs / sizeof(float)) + tdx];	

}

#endif