#include "gpu_vcd_sf_mp_k2.cuh"
#include "gpu_vcd_common.cuh"
#include "gpu_common.cuh"

__device__ void vcd_sf_mp_k2_load_ch_k(float* input_s, size_t inputptcs,
										const float* input, size_t inputptc,
										vcd_params params, int major, int minor)
{
	int loopx, loopy = 0;

	int maxrad = vcd_get_maxrad_k(params);
	int sx = blockIdx.x * blockDim.x + maxrad / 2 - 1;
	int sy = blockIdx.y * blockDim.y + maxrad / 2 - 1;

	// determine alignment of base address
	int bankblk = get_gmem_alignment_diff(sx * sizeof(float), major, minor);
	bankblk /= sizeof(float);

	// determine total load length
	if (bankblk == 0)
		loopx = (int)ceil(((float)(blockDim.x + 2))/(float)blockDim.x); // threads are alternatingly load each element
	else
		loopx = 1 + (int)ceil(((float)(blockDim.x + 2))/(float)blockDim.x);

	loopy = (int)ceil(((float)(blockDim.y + 2))/(float)blockDim.y);
	
	for (int y = 0; y < loopy; y++)
	{
		for (int x = 0; x < loopx; x++)
		{
			int nsx = sx + x * blockDim.x + threadIdx.x - bankblk;
			int nsy = sy + y * blockDim.y + threadIdx.y;

			int endx = sx + blockDim.x + 2;
			int endy = sy + blockDim.y + 2;

			if (nsx < endx && nsy < endy && 
				nsx >= (maxrad / 2 - 1) && nsy >= (maxrad / 2 - 1) && 
				nsx < ((maxrad + params.width) / 2 + 1) && nsy < ((maxrad + params.width) / 2 + 1) &&
				nsx >= sx && nsy >= sy)
			{
				int tx = x * blockDim.x + threadIdx.x;
				int ty = y * blockDim.y + threadIdx.y;

				input_s[ty * (inputptcs / sizeof(float)) + tx] = input[nsy * (inputptc / sizeof(float)) + nsx];
			}
		}
	}

}

__device__ void vcd_sf_mp_ch_get_offset(int* left, int* right, int* top, int* btm, int channel, int filter)
{
	int color = fc(filter, 0, 0);
	
	if (channel == COLOR_GREEN_RED)
	{
		// determine left/right
		if (color == COLOR_GREEN_RED || color == COLOR_BLUE)
		{
			*left = -1;
			*right = 0;
		} else // color == COLOR_GREEN_BLUE || color == COLOR_RED
		{
			*left = 0;
			*right = 1;
		}

		// determine up/down
		if (color == COLOR_GREEN_RED || color == COLOR_RED)
		{
			*top = -1;
			*btm = 0;
		} else 
		{
			*top = 0;
			*btm = 1;
		}

	} else if (channel == COLOR_GREEN_BLUE)
	{
		// determine left/right
		if (color == COLOR_GREEN_BLUE || color == COLOR_RED)
		{
			*left = -1;
			*right = 0;
		} else // color == COLOR_GREEN_RED || color == COLOR_BLUE
		{
			*left = 0;
			*right = 1;
		}

		// determine up/down
		if (color == COLOR_GREEN_BLUE || color == COLOR_BLUE)
		{
			*top = -1;
			*btm = 0;
		} else 
		{
			*top = 0;
			*btm = 1;
		}
	} else if (channel == COLOR_RED)
	{
		// determine left/right
		if (color == COLOR_RED || color == COLOR_GREEN_BLUE)
		{
			*left = -1;
			*right = 0;
		} else // color == COLOR_GREEN_RED || color == COLOR_BLUE
		{
			*left = 0;
			*right = 1;
		}

		// determine up/down
		if (color == COLOR_RED || color == COLOR_GREEN_RED)
		{
			*top = -1;
			*btm = 0;
		} else // color == COLOR_GREEN_BLUE || color == COLOR_BLUE
		{
			*top = 0;
			*btm = 1;
		}

	} else if (channel == COLOR_BLUE)
	{
		// determine left/right
		if (color == COLOR_BLUE || color == COLOR_GREEN_RED)
		{
			*left = -1;
			*right = 0;
		} else // color == COLOR_GREEN_BLUE || color == COLOR_RED
		{
			*left = 0;
			*right = 1;
		}

		// determine up/down
		if (color == COLOR_BLUE || color == COLOR_GREEN_BLUE)
		{
			*top = -1;
			*btm = 0;
		} else 
		{
			*top = 0;
			*btm = 1;
		}
	}
}

__device__ void vcd_sf_mp_rb_at_g_k(float* outch_g, size_t outchptc, const float* centerch_s, const float* neighbor_g_s, const float* neighbor_non_g_s, size_t chptcs, int channel, int direction, int filter, vcd_params params)
{
	int maxrad = vcd_get_maxrad_k(params);
	int left, right, top, btm = 0;
	vcd_sf_mp_ch_get_offset(&left, &right, &top, &btm, channel, filter);

	int sx = threadIdx.x + 1;
	int sy = threadIdx.y + 1;

	int tx = blockIdx.x * blockDim.x + threadIdx.x + maxrad / 2;
	int ty = blockIdx.y * blockDim.y + threadIdx.y + maxrad / 2;

	if (tx < (params.width + maxrad) / 2 && ty < (params.height + maxrad) / 2)
	{
		float val = 0.0f;

		if (direction == VERTICAL)
		{
			val = centerch_s[sy * (chptcs / sizeof(float)) + sx] + (
				neighbor_non_g_s[(sy + top) * (chptcs / sizeof(float)) + sx] -
				neighbor_g_s[(sy + top) * (chptcs / sizeof(float)) + sx] +
				neighbor_non_g_s[(sy + btm) * (chptcs / sizeof(float)) + sx] - 
				neighbor_g_s[(sy + btm) * (chptcs / sizeof(float)) + sx]) / 2.0f;
		} else if (direction == HORIZONTAL)
		{
			val = centerch_s[sy * (chptcs / sizeof(float)) + sx] + (
				neighbor_non_g_s[sy * (chptcs / sizeof(float)) + sx + left] -
				neighbor_g_s[sy * (chptcs / sizeof(float)) + sx + left] +
				neighbor_non_g_s[sy * (chptcs / sizeof(float)) + sx + right] - 
				neighbor_g_s[sy * (chptcs / sizeof(float)) + sx + right]) / 2.0f;
		}		

		outch_g[ty * (outchptc / sizeof(float)) + tx] = val;
	}
}

__device__ void vcd_sf_mp_rb_br_k(float* outch_g, size_t outchptc, const float* centerch_s, const float* neighbor_g_s, const float* neighbor_non_g_s, size_t chptcs, int channel, int filter, vcd_params params)
{
	int maxrad = vcd_get_maxrad_k(params);
	int left, right, top, btm = 0;
	vcd_sf_mp_ch_get_offset(&left, &right, &top, &btm, channel, filter);

	int sx = threadIdx.x + 1;
	int sy = threadIdx.y + 1;

	int tx = blockIdx.x * blockDim.x + threadIdx.x + maxrad / 2;
	int ty = blockIdx.y * blockDim.y + threadIdx.y + maxrad / 2;

	if (tx < (params.width + maxrad) / 2 && ty < (params.height + maxrad) / 2)
	{

		float val = 0.0f;

		val = centerch_s[sy * (chptcs / sizeof(float)) + sx] + (
			(neighbor_non_g_s[(sy+top) * (chptcs / sizeof(float)) + (sx+left)] - neighbor_g_s[(sy+top) * (chptcs / sizeof(float)) + (sx+left)]) +
			(neighbor_non_g_s[(sy+top) * (chptcs / sizeof(float)) + (sx+right)] - neighbor_g_s[(sy+top) * (chptcs / sizeof(float)) + (sx+right)]) +
			(neighbor_non_g_s[(sy+btm) * (chptcs / sizeof(float)) + (sx+left)] - neighbor_g_s[(sy+btm) * (chptcs / sizeof(float)) + (sx+left)]) +
			(neighbor_non_g_s[(sy+btm) * (chptcs / sizeof(float)) + (sx+right)] - neighbor_g_s[(sy+btm) * (chptcs / sizeof(float)) + (sx+right)])
			) / 4.0f;
		
		outch_g[ty * (outchptc / sizeof(float)) + tx] = val;
	}
}

__global__ void calculate_rest_k(float* grr, float* grb, float* gbr, float* gbb, float* br, float* rb, 
									const float* grg, const float* gbg, const float* rg, const float* bg, const float* rr, const float* bb, 
									size_t chptc, size_t chptcs, int filter, vcd_params params, int major, int minor)
{
	int color = fc(filter, 0, 0);
	int maxrad = vcd_get_maxrad_k(params);
	int left, right = 0;
	int sz = (chptcs / sizeof(float) * (blockDim.y + 2));

	// shared
	extern __shared__ float rr_s[]; 
	float* rg_s = &rr_s[sz];
	float* bb_s = &rg_s[sz];
	float* bg_s = &bb_s[sz];
	float* centerch_s = &bg_s[sz]; // G<R/B>G channel

	// load RR, RG, BB, BG
	vcd_sf_mp_k2_load_ch_k(rr_s, chptcs, rr, chptc, params, major, minor);
	__syncthreads();
	vcd_sf_mp_k2_load_ch_k(rg_s, chptcs, rg, chptc, params, major, minor);
	__syncthreads();
	vcd_sf_mp_k2_load_ch_k(bb_s, chptcs, bb, chptc, params, major, minor);
	__syncthreads();
	vcd_sf_mp_k2_load_ch_k(bg_s, chptcs, bg, chptc, params, major, minor);
	__syncthreads();
	
	// load GRG channel
	vcd_sf_mp_k2_load_ch_k(centerch_s, chptcs, grg, chptc, params, major, minor);
	__syncthreads();
	
	// calculate and store GRR and GRB
	vcd_sf_mp_rb_at_g_k(grr, chptc, centerch_s, rg_s, rr_s, chptcs, COLOR_GREEN_RED, HORIZONTAL, filter, params);

	vcd_sf_mp_rb_at_g_k(grb, chptc, centerch_s, bg_s, bb_s, chptcs, COLOR_GREEN_RED, VERTICAL, filter, params);

	__syncthreads();

	// load GBG channel
	vcd_sf_mp_k2_load_ch_k(centerch_s, chptcs, gbg, chptc, params, major, minor);
	__syncthreads();

	// calculate and store GBB and GBR
	vcd_sf_mp_rb_at_g_k(gbb, chptc, centerch_s, bg_s, bb_s, chptcs, COLOR_GREEN_BLUE, HORIZONTAL, filter, params);

	vcd_sf_mp_rb_at_g_k(gbr, chptc, centerch_s, rg_s, rr_s, chptcs, COLOR_GREEN_BLUE, VERTICAL, filter, params);

	// calculate and store BR
	vcd_sf_mp_rb_br_k(rb, chptc, rg_s, bg_s, bb_s, chptcs, COLOR_RED, filter, params);

	// calculate and store RB
	vcd_sf_mp_rb_br_k(br, chptc, bg_s, rg_s, rr_s, chptcs, COLOR_BLUE, filter, params);

}

void calculate_rest(float* grr, float* grb, float* gbr, float *gbb, float* br, float* rb, 
					   const float* grg, const float* gbg, const float* rg, const float* bg, const float* rr, const float* bb, 
					   size_t chptc, int filter, vcd_params params, cudaDeviceProp prop, vcd_sf_mp_kernel_times *times)

{
	dim3 dimBlock, dimGrid;
	size_t smemsz, smemptc = 0;

	if (prop.major == 2)
	{
		dimBlock.x = 32;
	} else
	{
		dimBlock.x = 16;
	}
	dimBlock.y = 16;

	smemptc = (dimBlock.x + 2) * sizeof(float);
	smemsz = smemptc * 5 * (dimBlock.y + 2);

	dimGrid.x = (int)ceil((float)params.width/(2.0f * dimBlock.x));
	dimGrid.y = (int)ceil((float)params.height/(2.0f * dimBlock.y));

	float kerntime = 0.0f;
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// calculate for GRx channel
	calculate_rest_k<<<dimGrid, dimBlock, smemsz>>>(grr, grb, gbr, gbb, br, rb, grg, gbg, rg, bg, rr, bb, chptc, smemptc, filter, params, prop.major, prop.minor);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&kerntime, start, stop);

	times->convert_output = kerntime;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}
