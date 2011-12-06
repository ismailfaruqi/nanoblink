#include "gpu_vcd_sf_mp_k1.cuh"
#include "gpu_common.cuh"
#include "gpu_vcd_common.cuh"
#include "vcd_common.h"

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
	cudaDeviceProp prop, vcd_sf_mp_kernel_times* times)
{
	float timer, timeb = 0.0f;
	cudaStream_t streams[1];

	// create stream
	for (int i = 0; i < 1; i++)
		cudaStreamCreate(&streams[i]);

	// stream[0]: calculate G on R pixel, diagonal
	interpolate_g_dg_neg(outg, outg, outptc, tempvarh, tempvarv, tempvarbh, tempvarbv, tempvarptc, 
		tempgh, tempgv, tempgd, tempgptc, input, inputptc, e, eptc, COLOR_RED, filter, params, prop, 0, &timer);

	// stream[1]: calculate G on B pixel, diagonal
	interpolate_g_dg_neg(outg, outg, outptc, tempvarh, tempvarv, tempvarbh, tempvarbv, tempvarptc, 
		tempgh, tempgv, tempgd, tempgptc, input, inputptc, e, eptc, COLOR_BLUE, filter, params, prop, streams[0], &timeb);

	// destroy stream
	for (int i = 0; i < 1; i++)
		cudaStreamDestroy(streams[i]);
	
	// add time
	times->calc_var_neg = timer + timeb;
}

void copy_to_channel(float* outr, float* outg, float* outb, size_t outptc,
	const float* input, size_t inputptc,
	int filter, vcd_params params, cudaStream_t stream)
{
	dim3 blockDim, gridDim;

	blockDim.x = 16;
	blockDim.y = 16;

	gridDim.x = (int)ceil((float)params.width/(float)blockDim.x);
	gridDim.y = (int)ceil((float)params.width/(float)blockDim.x);

	size_t smemptc, smemsz = 0;
	int mod = 0;
	smemptc = blockDim.x * sizeof(float);
	mod =  smemptc % 64;
	if (mod > 0)
		smemptc += (64 - mod);
	smemsz = smemptc * (blockDim.y);

	copy_to_channel_k<<<gridDim, blockDim, smemsz, stream>>>(outr, outg, outb, outptc, input, inputptc, smemptc, filter, params);
}

__global__ void copy_to_channel_k(float* outr, float* outg, float* outb, size_t outptc,
	const float* input, size_t inputptc,
	size_t smemptc,
	int filter, vcd_params params)
{
	int sx, sy, tx, ty, cx, cy;
	extern __shared__ float rch[];
	float* bch = &rch[blockDim.y / 2 * (smemptc / sizeof(float)) + blockDim.x];
	float* gch = &bch[blockDim.y / 2 * (smemptc / sizeof(float)) + blockDim.x];
	
	// load from input
	
	// store to shared memory

	// load from shared memory
	sx = threadIdx.x;
	sy = threadIdx.y;

	tx = blockDim.x;
	ty = blockDim.y;
}

// load temp green per channel
__device__ void vcd_sf_mp_load_tempg_k(float* tgt, size_t tgtptc, const float* tempg, size_t tempgptc, 
									   int bidx, int bidy, int channel, int filter, vcd_params params)
{
	int sx, sy, tx, ty,loopx, loopy, nsx, nsy, color, endx, endy, cx, cy = 0;
	int maxrad = max(params.window_radius + params.temp_green_radius, params.e_radius); // input radius

	// base address for thread block
	sx = bidx * blockDim.x; // maxrad is apparently subject to be variable
	sy = bidy * 2 * blockDim.y;

	loopx = (int)ceil(((float)(blockDim.x + params.window_radius / 2))/(float)blockDim.x); // threads are alternatingly load each element
	loopy = (int)ceil(((float)(blockDim.y + params.window_radius / 2))/(float)blockDim.y);

	for (int y = 0; y < loopy; y++)
	{
		for (int x = 0; x < loopx; x++)
		{
			nsx = sx + x * blockDim.x + threadIdx.x;
			nsy = sy + 2 * (y * blockDim.y + threadIdx.y);

			endx = sx + blockDim.x + params.window_radius / 2;
			endy = sy + 2 * blockDim.y + params.window_radius;

			cx = nsx;
			cy = nsy;

			color = fc(filter, cx, cy);

			// adjust only y
			if (channel == COLOR_RED)
			{
				if (color == COLOR_GREEN_BLUE || color == COLOR_BLUE)
				{
					nsy--;
					cy--;
				}

			} else if (channel == COLOR_BLUE)
			{
				if (color == COLOR_GREEN_RED || color == COLOR_RED)
				{
					nsy++;
					cy++;
				}
			}

			if (nsx < endx && nsy < endy &&
				nsx < (params.width + params.window_radius) / 2 && nsy < (params.height + params.window_radius)
				)
			{
				tx = x * blockDim.x + threadIdx.x;
				ty = y * blockDim.y + threadIdx.y;

				tgt[ty * (tgtptc / sizeof(float)) + tx] = tempg[nsy * (tempgptc / sizeof(float)) + nsx];
			}
		
		}
	}
}

void interpolate_g_dg_neg(float* out, float* outg, size_t outptc,
	float* tempvarh, float* tempvarv, float* tempvarbh, float* tempvarbv, size_t tempvarptc,
	const float* tempgh, const float* tempgv, const float* tempgd, size_t tempgptc,
	const float* input, size_t inputptc,
	const int* e, size_t eptc,
	int channel, int filter, vcd_params params, cudaDeviceProp prop, cudaStream_t stream, float* time)
{
	dim3 dimBlock, dimGrid;
	smeminfo_interpolate_g_neg info;

	if (prop.major == 2)
	{
		dimBlock.x = 32;
		dimBlock.y = 16;
	}
	else // 1.x
	{
		dimBlock.x = 16;
		dimBlock.y = 16;
	}

	cudaEvent_t start, stop;
	float kerntime = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// shared memory size
	size_t smemsz = 0;
	
	// init smeminfo
	smeminfo_interpolate_g_neg_init(&info);
	info.inputptcs = ((dimBlock.x + 2 * params.window_radius / 2) * sizeof(float));
	info.outputgptcs = ((dimBlock.x + params.window_radius / 2) * sizeof(float));
	info.tempgptcs = ((dimBlock.x + params.window_radius / 2) * sizeof(float));
	info.varptcs = (dimBlock.x * sizeof(float));
		
	smemsz += (info.inputptcs * (dimBlock.y + 2 * params.window_radius / 2));
	smemsz += (info.outputgptcs * (dimBlock.y + params.window_radius / 2));
	info.tempglen = get_smem_pitch(info.tempgptcs * (dimBlock.y + params.window_radius / 2));
	smemsz += (3 * info.tempglen);
	smemsz += (4 * info.varptcs * dimBlock.y);
		
	// down-sweep
	int ymax = (int)ceil((float)(params.height) / (float)(dimBlock.y * 2));
	for (int y = 0; y < ymax; y++)
	{
		// determine grid dimension
		int xmax = (int)ceil((float)params.width / (float)(dimBlock.x * 2));
		int maxgrid = min(y+1,xmax);
		dimGrid.x = maxgrid;
		
		// launch kernel
		cudaEventRecord(start,stream);
		interpolate_g_dg_neg_k<<<dimGrid, dimBlock, smemsz, stream>>>(out, outg, outptc, tempvarh, tempvarv, tempvarbh, tempvarbv, tempvarptc,
			tempgh, tempgv, tempgd, tempgptc, input, inputptc, e, eptc, channel, DOWN_SWEEP, y, filter, params, info, prop.major, prop.minor);
		cudaEventRecord(stop,stream);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&kerntime,start,stop);
		*time += kerntime;
	}

	// right-sweep
	int xmax = (int)ceil((float)(params.width - dimBlock.x * 2) / (dimBlock.x * 2));
	for (int x = 0; x < xmax; x++)
	{
		int ymax = (int)ceil((float)(params.height) / (float)(dimBlock.y * 2));
		int maxgrid = min(xmax - x, ymax);
		dimGrid.x = maxgrid;

		// launch kernel
		cudaEventRecord(start,stream);
		interpolate_g_dg_neg_k<<<dimGrid, dimBlock, smemsz, stream>>>(out, outg, outptc, tempvarh, tempvarv, tempvarbh, tempvarbv, tempvarptc,
			tempgh, tempgv, tempgd, tempgptc, input, inputptc, e, eptc, channel, RIGHT_SWEEP, x, filter, params, info, prop.major, prop.minor);
		cudaEventRecord(stop,stream);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&kerntime,start,stop);
		*time += kerntime;
	}

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

// load input per channel in positive direction
__device__ void vcd_sf_mp_load_input_ch_neg_k(float* tgt, size_t tgtptc, const float* input, size_t inputptc, int channel, int filter, vcd_params params, 
											  int bidx, int bidy, int major, int minor)
{
	int sx, sy, tx, ty,loopx, loopy, nsx, nsy, color, bankblk, endx, endy, cx, cy = 0;
	int maxrad = max(params.window_radius + params.temp_green_radius, params.e_radius); // input radius

	// base address for thread block
	sx = bidx * 2 * blockDim.x + params.temp_green_radius;
	sy = bidy * 2 * blockDim.y + params.temp_green_radius;

	// determine alignment of base address
	bankblk = get_gmem_alignment_diff(sx * sizeof(float), major, minor);
	bankblk /= sizeof(float);

	// determine total load length
	if (bankblk == 0)
		loopx = (int)ceil(((float)(2 * blockDim.x + 2 * params.window_radius))/(float)blockDim.x); // threads are alternatingly load each element
	else
		loopx = 1 + (int)ceil(((float)(2 * blockDim.x + 2 * params.window_radius))/(float)blockDim.x);

	loopy = (int)ceil(((float)(blockDim.y + params.window_radius))/(float)blockDim.y);

	for (int y = 0; y < loopy; y++)
	{
		for (int x = 0; x < loopx; x++)
		{
			nsx = sx + x * blockDim.x + threadIdx.x - bankblk;
			nsy = sy + 2 * (y * blockDim.y + threadIdx.y);

			endx = sx + 2 * blockDim.x + 2 * params.window_radius;
			endy = sy + 2 * blockDim.y + 2 * params.window_radius;

			cx = nsx + params.window_radius;
			cy = nsy + params.window_radius;

			color = fc(filter, cx, cy);

			// adjust only y
			if (channel == COLOR_RED)
			{
				if (color == COLOR_GREEN_BLUE || color == COLOR_BLUE)
				{
					nsy--;
					cy--;
				}

			} else if (channel == COLOR_BLUE)
			{
				if (color == COLOR_GREEN_RED || color == COLOR_RED)
				{
					nsy++;
					cy++;
				}
			}

			// we need to check color of the thread once more
			color = fc(filter, cx, cy);

			if (color == channel)
			{
				if (nsx < endx && nsy < endy && nsx >= params.temp_green_radius && nsy >= params.temp_green_radius && 
					nsx < (params.width + maxrad + params.window_radius) && nsy < (params.height + maxrad + params.window_radius) &&
					nsx >= sx && nsy >= sy)
				{
					tx = (x * blockDim.x + threadIdx.x - bankblk) / 2;
					ty = y * blockDim.y + threadIdx.y;

					tgt[ty * (tgtptc / sizeof(float)) + tx] = input[nsy * (inputptc / sizeof(float)) + nsx];
				}
			}
		}
	}
}

// a function for testing vcd_sf_mp_load_input_ch. It copies a channel in shared memory to an output channel.
__device__ void vcd_sf_mp_test_load_input_ch_neg_k(float* out, size_t outptc, const float* input, size_t inputptc, int channel, int filter, vcd_params params,
												   int bidx, int bidy)
{
	int sx, sy, tx, ty, color = 0;

	sx = threadIdx.x + params.window_radius / 2;
	sy = threadIdx.y + params.window_radius / 2;

	tx = 2 * (bidx * blockDim.x + threadIdx.x);
	ty = 2 * (bidy * blockDim.y + threadIdx.y);

	// adjust position
	color = fc(filter, tx, ty);

	if (channel == COLOR_RED)
	{
		if (color == COLOR_GREEN_BLUE || color == COLOR_BLUE)
		{
			ty--;
		}

		if (color == COLOR_GREEN_RED || color == COLOR_RED)
		{
			tx--;
		}

	} else if (channel == COLOR_BLUE)
	{
		if (color == COLOR_GREEN_RED || color == COLOR_RED)
		{
			ty++;
		}

		if (color == COLOR_GREEN_BLUE || color == COLOR_RED)
		{
			tx++;
		}
	}

	if (tx < params.width && ty < params.height)
		out[ty * (outptc / sizeof(float)) + tx] = input[sy * (inputptc / sizeof(float)) + sx];
}

// a function for testing vcd_sf_mp_load_input_ch. It copies a channel in shared memory to an output channel.
__device__ void vcd_sf_mp_test_load_tempg_neg_k(float* out, size_t outptc, const float* input, size_t inputptc, int channel, int filter, vcd_params params,
												   int bidx, int bidy)
{
	int sx, sy, tx, ty, color = 0;

	sx = threadIdx.x;
	sy = threadIdx.y;

	tx = 2 * (bidx * blockDim.x + threadIdx.x);
	ty = 2 * (bidy * blockDim.y + threadIdx.y);

	// adjust position
	color = fc(filter, tx, ty);

	if (channel == COLOR_RED)
	{
		if (color == COLOR_GREEN_BLUE || color == COLOR_BLUE)
		{
			ty--;
		}

		if (color == COLOR_GREEN_RED || color == COLOR_RED)
		{
			tx--;
		}

	} else if (channel == COLOR_BLUE)
	{
		if (color == COLOR_GREEN_RED || color == COLOR_RED)
		{
			ty++;
		}

		if (color == COLOR_GREEN_BLUE || color == COLOR_RED)
		{
			tx++;
		}
	}

	if (tx < params.width && ty < params.height)
		out[ty * (outptc / sizeof(float)) + tx] = input[sy * (inputptc / sizeof(float)) + sx];
}

// load input per channel in positive direction
__device__ void vcd_sf_mp_load_output_neg_k(float* tgt, size_t tgtptc, const float* src, size_t srcptc, int channel, int filter, vcd_params params, 
											  int bidx, int bidy, int major, int minor)
{
	int sx, sy, tx, ty,loopx, loopy, nsx, nsy, color, bankblk, endx, endy, cx, cy = 0;
	int maxrad = max(params.window_radius + params.temp_green_radius, params.e_radius); // input radius

	// base address for thread block
	sx = bidx * 2 * blockDim.x - params.window_radius;
	sy = bidy * 2 * blockDim.y - params.window_radius;

	// determine alignment of base address
	bankblk = get_gmem_alignment_diff(sx * sizeof(float), major, minor);
	bankblk /= sizeof(float);

	// determine total load length
	if (bankblk == 0)
		loopx = (int)ceil(((float)(2 * blockDim.x + params.window_radius))/(float)blockDim.x); // threads are alternatingly load each element
	else
		loopx = 1 + (int)ceil(((float)(2 * blockDim.x + params.window_radius))/(float)blockDim.x);

	loopy = (int)ceil(((float)(blockDim.y + params.window_radius / 2))/(float)blockDim.y);

	for (int y = 0; y < loopy; y++)
	{
		for (int x = 0; x < loopx; x++)
		{
			nsx = sx + x * blockDim.x + threadIdx.x - bankblk;
			nsy = sy + 2 * (y * blockDim.y + threadIdx.y);

			endx = sx + 2 * blockDim.x + params.window_radius;
			endy = sy + 2 * blockDim.y + params.window_radius;

			cx = nsx;
			cy = nsy;

			color = fc(filter, cx, cy);

			// adjust only y
			if (channel == COLOR_RED)
			{
				if (color == COLOR_GREEN_BLUE || color == COLOR_BLUE)
				{
					nsy--;
					cy--;
				}

			} else if (channel == COLOR_BLUE)
			{
				if (color == COLOR_GREEN_RED || color == COLOR_RED)
				{
					nsy++;
					cy++;
				}
			}

			// we need to check color of the thread once more
			color = fc(filter, cx, cy);

			if (color == channel)
			{
				if (nsx < endx && nsy < endy && 
					nsx < (params.width) && nsy < (params.height) &&
					nsx >= sx && nsy >= sy)
				{
					tx = (x * blockDim.x + threadIdx.x - bankblk) / 2;
					ty = y * blockDim.y + threadIdx.y;

					if (nsx < 0 || nsy < 0)
						tgt[ty * (tgtptc / sizeof(float)) + tx] = 0.0f;
					else
						tgt[ty * (tgtptc / sizeof(float)) + tx] = src[nsy * (srcptc / sizeof(float)) + nsx];
				}
			}
		}
	}
}

// load temp green per channel
__device__ void vcd_sf_mp_load_var_k(float* tgt, size_t tgtptc, const float* tempg, size_t tempgptc, int channel, int filter, vcd_params params,
										   int bidx, int bidy)
{
	int sx, sy, tx, ty, color = 0;
	
	// base address for thread block
	sx = bidx * blockDim.x + threadIdx.x; // maxrad is apparently subject to be variable
	sy = bidy * 2 * (blockDim.y + threadIdx.y);

	color = fc(filter, sx, sy);

	// adjust only y
	if (channel == COLOR_RED)
	{
		if (color == COLOR_GREEN_BLUE || color == COLOR_BLUE)
		{
			sy--;
		}

	} else if (channel == COLOR_BLUE)
	{
		if (color == COLOR_GREEN_RED || color == COLOR_RED)
		{
			sy++;
		}
	}

	if (sx < params.width / 2 && sy < params.height)
	{
		tx = threadIdx.x;
		ty = threadIdx.y;

		tgt[ty * (tgtptc / sizeof(float)) + tx] = tempg[sy * (tempgptc / sizeof(float)) + sx];
	}
		
}

// calculate variance in vertical OR horizontal
__device__ void vcd_sf_mp_calculate_var_k(float* var, float* inputch_s, size_t inputptcs, float* outputg_s, size_t outputgptcs, 
				const float* var_s, size_t varptcs, 
				const float* tempg_s, size_t tempgptcs, 
				int direction, int tdx, int tdy, int window_radius)
{
	int sx = tdx + window_radius / 2;
	int sy = tdy + window_radius / 2;

	if (direction == HORIZONTAL)
	{
		float res0 = inputch_s[sy * (inputptcs / sizeof(float)) + sx] - tempg_s[tdy * (tempgptcs / sizeof(float)) + tdx];
		float resm2 = inputch_s[sy * (inputptcs / sizeof(float)) + (sx - 1)] - outputg_s[sy * (outputgptcs / sizeof(float)) + (sx - 1)];
		float resm4 = inputch_s[sy * (inputptcs / sizeof(float)) + (sx - 2)] - outputg_s[sy * (outputgptcs / sizeof(float)) + (sx - 2)];
		
		float res = var_s[tdy * (varptcs / sizeof(float)) + tdx] + resm2 + resm4 + (res0 + resm2) / 2.0f + (resm2 + resm4) / 2.0f; // sigma
		res /= 9.0f; // 1/9 sigma

		// each of (d - sigma)^2
		res0 = (inputch_s[sy * (inputptcs / sizeof(float)) + sx] - tempg_s[tdy * (tempgptcs / sizeof(float)) + tdx] - res);
		res0 *= res0;
		float res2 = (inputch_s[sy * (inputptcs / sizeof(float)) + (sx + 1)] - tempg_s[tdy * (tempgptcs / sizeof(float)) + (tdx + 1)] - res);
		res2 *= res2;
		float res4 = (inputch_s[sy * (inputptcs / sizeof(float)) + (sx + 2)] - tempg_s[tdy * (tempgptcs / sizeof(float)) + (tdx + 2)] - res);
		res4 *= res4;
		resm2 = (inputch_s[sy * (inputptcs / sizeof(float)) + (sx - 1)] - outputg_s[sy * (outputgptcs / sizeof(float)) + (sx - 1)] - res);
		resm2 *= resm2;
		resm4 = (inputch_s[sy * (inputptcs / sizeof(float)) + (sx - 2)] - outputg_s[sy * (outputgptcs / sizeof(float)) + (sx - 2)] - res);
		resm4 *= resm4;

		// 1/9 * sigma (d - sigma)^2
		*var = (res0 + res2 + res4 + resm2 + resm4 + (res0 + res2) / 2.0f + (res2 + res4) / 2.0f + (res0 + resm2) / 2.0f + (resm2 + resm4) / 2.0f) / 9.0f;

	} else if (direction == VERTICAL)
	{
		float res0 = inputch_s[sy * (inputptcs / sizeof(float)) + sx] - tempg_s[tdy * (tempgptcs / sizeof(float)) + tdx];
		float resm2 = inputch_s[(sy - 1) * (inputptcs / sizeof(float)) + sx] - outputg_s[(sy - 1)* (outputgptcs / sizeof(float)) + sx];
		float resm4 = inputch_s[(sy - 2) * (inputptcs / sizeof(float)) + sx] - outputg_s[(sy - 2) * (outputgptcs / sizeof(float)) + sx];
		
		float res = var_s[tdy * (varptcs / sizeof(float)) + tdx] + resm2 + resm4 + (res0 + resm2) / 2.0f + (resm2 + resm4) / 2.0f; // sigma
		res /= 9.0f; // 1/9 sigma

		// each of (d - sigma)^2
		res0 = (inputch_s[sy * (inputptcs / sizeof(float)) + sx] - tempg_s[tdy * (tempgptcs / sizeof(float)) + tdx] - res);
		res0 *= res0;
		float res2 = (inputch_s[(sy + 1) * (inputptcs / sizeof(float)) + sx] - tempg_s[(tdy + 1) * (tempgptcs / sizeof(float)) + tdx] - res);
		res2 *= res2;
		float res4 = (inputch_s[(sy + 2) * (inputptcs / sizeof(float)) + sx] - tempg_s[(tdy + 2) * (tempgptcs / sizeof(float)) + tdx] - res);
		res4 *= res4;
		resm2 = (inputch_s[(sy - 1)* (inputptcs / sizeof(float)) + sx] - outputg_s[(sy - 1) * (outputgptcs / sizeof(float)) + sx] - res);
		resm2 *= resm2;
		resm4 = (inputch_s[(sy - 2) * (inputptcs / sizeof(float)) + sx] - outputg_s[(sy - 2) * (outputgptcs / sizeof(float)) + sx] - res);
		resm4 *= resm4;

		// 1/9 * sigma (d - sigma)^2
		*var = (res0 + res2 + res4 + resm2 + resm4 + (res0 + res2) / 2.0f + (res2 + res4) / 2.0f + (res0 + resm2) / 2.0f + (resm2 + resm4) / 2.0f) / 9.0f;

	}

}

// calculate variance in diagonal 
__device__ void vcd_sf_mp_calculate_vard_k(float* var, float* inputch_s, size_t inputptcs, float* outputg_s, size_t outputgptcs, 
				const float* vardh_s, const float* vardv_s, size_t varptcs, 
				const float* tempg_s, size_t tempgptcs, 
				int tdx, int tdy, int window_radius)
{
	float resh, resv, resm4, resm2, res0, res2, res4 = 0.0f;
	int sx = tdx + window_radius / 2;
	int sy = tdy + window_radius / 2;

	// horizontal element
	res0 = inputch_s[sy * (inputptcs / sizeof(float)) + sx] - tempg_s[tdy * (tempgptcs / sizeof(float)) + tdx];
	resm2 = inputch_s[sy * (inputptcs / sizeof(float)) + (sx - 1)] - outputg_s[sy * (outputgptcs / sizeof(float)) + (sx - 1)];
	resm4 = inputch_s[sy * (inputptcs / sizeof(float)) + (sx - 2)] - outputg_s[sy * (outputgptcs / sizeof(float)) + (sx - 2)];

	resh = vardh_s[tdy * (varptcs / sizeof(float)) + tdx] + resm2 + resm4 + (res0 + resm2) / 2.0f + (resm2 + resm4) / 2.0f; // sigma
	resh /= 9.0f; // 1/9 sigma

	// each of (d - sigma)^2
	res0 = (inputch_s[sy * (inputptcs / sizeof(float)) + sx] - tempg_s[tdy * (tempgptcs / sizeof(float)) + tdx] - resh);
	res0 *= res0;
	res2 = (inputch_s[sy * (inputptcs / sizeof(float)) + (sx + 1)] - tempg_s[tdy * (tempgptcs / sizeof(float)) + (tdx + 1)] - resh);
	res2 *= res2;
	res4 = (inputch_s[sy * (inputptcs / sizeof(float)) + (sx + 2)] - tempg_s[tdy * (tempgptcs / sizeof(float)) + (tdx + 2)] - resh);
	res4 *= res4;
	resm2 = (inputch_s[sy * (inputptcs / sizeof(float)) + (sx - 1)] - outputg_s[sy * (outputgptcs / sizeof(float)) + (sx - 1)] - resh);
	resm2 *= resm2;
	resm4 = (inputch_s[sy * (inputptcs / sizeof(float)) + (sx - 2)] - outputg_s[sy * (outputgptcs / sizeof(float)) + (sx - 2)] - resh);
	resm4 *= resm4;

	// 1/9 * sigma (d - sigma)^2
	resh = (res0 + res2 + res4 + resm2 + resm4 + (res0 + res2) / 2.0f + (res2 + res4) / 2.0f + (res0 + resm2) / 2.0f + (resm2 + resm4) / 2.0f) / 9.0f;

	// vertical element
	res0 = inputch_s[sy * (inputptcs / sizeof(float)) + sx] - tempg_s[tdy * (tempgptcs / sizeof(float)) + tdx];
	resm2 = inputch_s[(sy - 1) * (inputptcs / sizeof(float)) + sx] - outputg_s[(sy - 1)* (outputgptcs / sizeof(float)) + sx];
	resm4 = inputch_s[(sy - 2) * (inputptcs / sizeof(float)) + sx] - outputg_s[(sy - 2) * (outputgptcs / sizeof(float)) + sx];

	resv = vardv_s[tdy * (varptcs / sizeof(float)) + tdx] + resm2 + resm4 + (res0 + resm2) / 2.0f + (resm2 + resm4) / 2.0f; // sigma
	resv /= 9.0f; // 1/9 sigma

	// each of (d - sigma)^2
	res0 = (inputch_s[sy * (inputptcs / sizeof(float)) + sx] - tempg_s[tdy * (tempgptcs / sizeof(float)) + tdx] - resv);
	res0 *= res0;
	res2 = (inputch_s[(sy + 1) * (inputptcs / sizeof(float)) + sx] - tempg_s[(tdy + 1) * (tempgptcs / sizeof(float)) + tdx] - resv);
	res2 *= res2;
	res4 = (inputch_s[(sy + 2) * (inputptcs / sizeof(float)) + sx] - tempg_s[(tdy + 2) * (tempgptcs / sizeof(float)) + tdx] - resv);
	res4 *= res4;
	resm2 = (inputch_s[(sy - 1)* (inputptcs / sizeof(float)) + sx] - outputg_s[(sy - 1) * (outputgptcs / sizeof(float)) + sx] - resv);
	resm2 *= resm2;
	resm4 = (inputch_s[(sy - 2) * (inputptcs / sizeof(float)) + sx] - outputg_s[(sy - 2) * (outputgptcs / sizeof(float)) + sx] - resv);
	resm4 *= resm4;

	// 1/9 * sigma (d - sigma)^2
	resv = (res0 + res2 + res4 + resm2 + resm4 + (res0 + res2) / 2.0f + (res2 + res4) / 2.0f + (res0 + resm2) / 2.0f + (resm2 + resm4) / 2.0f) / 9.0f;

	// write output
	*var = (resh + resv) / 2.0f;
}

__device__ void vcd_sf_mp_store_outg_k(float* outg, size_t outgptc, const float* outputg_s, size_t outputgptcs, int channel, int filter, vcd_params params, int bidx, int bidy)
{
	int sx, sy, tx, ty, ntx, color = 0;

	tx = 2 * bidx * blockDim.x;
	ty = 2 * (bidy * blockDim.y + threadIdx.y);

	for (int x = 0; x < 2; x++)
	{
		ntx = tx + x * blockDim.x + threadIdx.x;
		
		// adjust position
		color = fc(filter, ntx, ty);

		if (channel == COLOR_RED)
		{
			if (color == COLOR_GREEN_BLUE || color == COLOR_BLUE)
			{
				ty--;
			}
			
		} else if (channel == COLOR_BLUE)
		{
			if (color == COLOR_GREEN_RED || color == COLOR_RED)
			{
				ty++;
			}			
		}

		// because we will do strided access, check color again
		color = fc(filter, ntx, ty);

		sx = (x * blockDim.x + threadIdx.x) / 2 + params.window_radius / 2;
		sy = threadIdx.y + params.window_radius / 2;

		if (channel == color && ntx < params.width && ty < params.height)
			outg[ty * (outgptc / sizeof(float)) + ntx] = outputg_s[sy * (outputgptcs / sizeof(float)) + sx];
	}

	
}

__global__ void interpolate_g_dg_neg_k(
	float* out, float* outg, size_t outptc,
	float* tempvarh, float* tempvarv, float* tempvarbh, float* tempvarbv, size_t tempvarptc,
	const float* tempgh, const float* tempgv, const float* tempgd, size_t tempgptc,
	const float* input, size_t inputptc,
	const int* e, size_t eptc,
	int channel, int direction, int idx, int filter, vcd_params params, smeminfo_interpolate_g_neg smeminfo,
	int major, int minor)
{
	int bdx, bdy = 0;
	
	// map block index, always do it first
	if (direction == DOWN_SWEEP)
	{
		bdx = blockIdx.x;
		bdy = idx - blockIdx.x;
	} else if (direction == RIGHT_SWEEP)
	{
		bdx = idx + blockIdx.x + 1;
		bdy = (int)ceil(float(params.height) / float(blockDim.y * 2)) - blockIdx.x - 1;
	} else if (direction == PARALLEL)
	{
		bdx = blockIdx.x;
		bdy = blockIdx.y;
	}

	extern __shared__ float inputch_s[];
	float* outputg_s = &inputch_s[(blockDim.y + 2 * params.window_radius / 2) * (smeminfo.inputptcs / sizeof(float))];
	float* tempgh_s = &outputg_s[(blockDim.y + params.window_radius / 2) * (smeminfo.outputgptcs / sizeof(float))];
	float* tempgv_s = &tempgh_s[smeminfo.tempglen / sizeof(float)];
	float* tempgd_s = &tempgv_s[smeminfo.tempglen / sizeof(float)];
	float* varh_s = &tempgd_s[smeminfo.tempglen / sizeof(float)];
	float* varv_s = &varh_s[blockDim.y * (smeminfo.varptcs / sizeof(float))];
	float* vardh_s = &varv_s[blockDim.y * (smeminfo.varptcs / sizeof(float))];
	float* vardv_s = &vardh_s[blockDim.y * (smeminfo.varptcs / sizeof(float))];
	
	// load input
	vcd_sf_mp_load_input_ch_neg_k(inputch_s, smeminfo.inputptcs, input, inputptc, channel, filter, params, bdx, bdy, major, minor);
	__syncthreads();

	// load output. Pad them with zero in the left and top apron
	vcd_sf_mp_load_output_neg_k(outputg_s, smeminfo.outputgptcs, out, outptc, channel, filter, params, bdx, bdy, major, minor);
	__syncthreads();

	// load temporary greens, to avoid uncoalesced access on global memory
	vcd_sf_mp_load_tempg_k(tempgh_s, smeminfo.tempgptcs, tempgh, tempgptc, bdx, bdy, channel, filter, params);
	__syncthreads();
	vcd_sf_mp_load_tempg_k(tempgv_s, smeminfo.tempgptcs, tempgv, tempgptc, bdx, bdy, channel, filter, params);
	__syncthreads();
	vcd_sf_mp_load_tempg_k(tempgd_s, smeminfo.tempgptcs, tempgd, tempgptc, bdx, bdy, channel, filter, params);
	__syncthreads();

	//vcd_sf_mp_test_load_tempg_neg_k(out, outptc, tempgh_s, smeminfo.tempgptcs, channel, filter, params, bdx, bdy);
	//__syncthreads();

	// load variances, to avoid uncoalesced access on global memory
	vcd_sf_mp_load_var_k(varh_s, smeminfo.varptcs, tempvarh, tempvarptc, channel, filter, params, bdx, bdy);
	__syncthreads();

	vcd_sf_mp_load_var_k(varv_s, smeminfo.varptcs, tempvarv, tempvarptc, channel, filter, params, bdx, bdy);
	__syncthreads();

	vcd_sf_mp_load_var_k(vardh_s, smeminfo.varptcs, tempvarbh, tempvarptc, channel, filter, params, bdx, bdy);
	__syncthreads();

	vcd_sf_mp_load_var_k(vardv_s, smeminfo.varptcs, tempvarbv, tempvarptc, channel, filter, params, bdx, bdy);
	__syncthreads();

	// calculate variance	

	// use only the first warp (fermi) or half warp ( < fermi) of the thread. Assume that blockDim.x >= blockDim.y
	
	// calculate variance element in down sweep direction
	for (int i = 0; i < blockDim.y; i++)
	{
		float varh, varv, vard = 0.0f; // loopx will hold horizontal calc. result. loopy will hold vertical calc. result

		int tdx = threadIdx.x;
		int tdy = i - threadIdx.x;

		if (tdx >= 0 && tdx <= i && tdy >= 0 && tdy <= i && threadIdx.y == 0)
		{

			vcd_sf_mp_calculate_var_k(&varh, inputch_s, smeminfo.inputptcs, outputg_s, smeminfo.outputgptcs, 
					varh_s, smeminfo.varptcs, 
					tempgh_s, smeminfo.tempgptcs, 
					HORIZONTAL, tdx, tdy, params.window_radius);

			vcd_sf_mp_calculate_var_k(&varv, inputch_s, smeminfo.inputptcs, outputg_s, smeminfo.outputgptcs, 
					varv_s, smeminfo.varptcs, 
					tempgv_s, smeminfo.tempgptcs, 
					VERTICAL, tdx, tdy, params.window_radius);

			vcd_sf_mp_calculate_vard_k(&vard, inputch_s, smeminfo.inputptcs, outputg_s, smeminfo.outputgptcs, 
					vardh_s, vardv_s, smeminfo.varptcs, 
					tempgd_s, smeminfo.tempgptcs, 
					tdx, tdy, params.window_radius);

			vcd_sf_mp_determinte_g_k(outputg_s, smeminfo.outputgptcs,
				tempgh_s, tempgv_s, tempgd_s, smeminfo.tempgptcs, smeminfo.tempglen,
				varh, varv, vard, tdx, tdy, params.window_radius);
		}

		__syncthreads();
	}

	// calculate variance element in right sweep direction
	for (int i = 1; i < blockDim.x; i++)
	{

		float varh, varv, vard = 0.0f; // loopx will hold horizontal calc. result. loopy will hold vertical calc. result

		int tdx = threadIdx.x + i;
		int tdy = blockDim.y - threadIdx.x - 1;

		if (tdx >= 0 && tdx < blockDim.x && tdy >= 0 && tdy < blockDim.y &&
			(tdx + tdy) == (blockDim.y + i - 1) && threadIdx.y == 0)
		{
			vcd_sf_mp_calculate_var_k(&varh, inputch_s, smeminfo.inputptcs, outputg_s, smeminfo.outputgptcs, 
					varh_s, smeminfo.varptcs, 
					tempgh_s, smeminfo.tempgptcs, 
					HORIZONTAL, tdx, tdy, params.window_radius);

			vcd_sf_mp_calculate_var_k(&varv, inputch_s, smeminfo.inputptcs, outputg_s, smeminfo.outputgptcs, 
					varv_s, smeminfo.varptcs, 
					tempgv_s, smeminfo.tempgptcs, 
					VERTICAL, tdx, tdy, params.window_radius);

			vcd_sf_mp_calculate_vard_k(&vard, inputch_s, smeminfo.inputptcs, outputg_s, smeminfo.outputgptcs, 
					vardh_s, vardv_s, smeminfo.varptcs, 
					tempgd_s, smeminfo.tempgptcs, 
					tdx, tdy, params.window_radius);

			vcd_sf_mp_determinte_g_k(outputg_s, smeminfo.outputgptcs,
				tempgh_s, tempgv_s, tempgd_s, smeminfo.tempgptcs, smeminfo.tempglen,
				varh, varv, vard, tdx, tdy, params.window_radius);

			
		}

		__syncthreads();
	}
	

	// store output G
	vcd_sf_mp_store_outg_k(outg, outptc, outputg_s, smeminfo.outputgptcs, channel, filter, params, bdx, bdy);
}

void interpolate_g_dg_pos(float* out, size_t outptc,
	float* tempvarh, float* tempvarv, float* tempvarbh, float* tempvarbv, size_t tempvarptc,
	const float* tempgh, const float* tempgv, const float* tempgd, size_t tempgptc,
	const float* input, size_t inputptc,
	const int* e, size_t eptc,
	int filter, vcd_params params, cudaDeviceProp prop, vcd_sf_mp_kernel_times* times)
{
	dim3 dimBlock, dimGrid;
	smeminfo_interpolate_g_pos info;

	float kerntime = 0.0f;
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	if (prop.major == 2)
		dimBlock.x = 32;
	else
		dimBlock.x = 16;
	dimBlock.y = 16;

	// shared memory size
	size_t smemsz = 0;
	
	// init smeminfo
	smeminfo_interpolate_g_pos_init(&info);
	info.inputptcs = get_smem_pitch((dimBlock.x + params.window_radius / 2) * sizeof(float)); // pitch for R channel and B channel
	info.tempgptcs = get_smem_pitch((dimBlock.x + params.window_radius / 2) * sizeof(float)); // pitch for temp G channel 
	
	smemsz += (info.inputptcs * (dimBlock.y + params.window_radius / 2)); // allocate for R and B channel
	smemsz += (info.tempgptcs * (dimBlock.y + params.window_radius / 2)); // allocate for temp G channel

	dimGrid.x = (int)ceil((float)(params.width / 2) / (float)dimBlock.x);
	dimGrid.y = (int)ceil((float)(params.height / 2) / (float)dimBlock.y);

	cudaEventRecord(start, 0);
	interpolate_g_dg_pos_k<<<dimGrid, dimBlock, smemsz>>>(out, outptc, tempvarh, tempvarv, tempvarbh, tempvarbv, tempvarptc,
			tempgh, tempgv, tempgd, tempgptc, input, inputptc, e, eptc, filter, params, info, prop.major, prop.minor);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&kerntime, start, stop);

	times->calc_temp_var_pos = kerntime;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	
}

// load input per channel in positive direction
__device__ void vcd_sf_mp_load_input_ch_pos_k(float* tgt, size_t tgtptc, const float* input, size_t inputptc, int channel, int filter, vcd_params params, int major, int minor)
{
	int sx, sy, tx, ty,loopx, loopy, nsx, nsy, color, bankblk, endx, endy, cx, cy = 0;
	int maxrad = max(params.window_radius + params.temp_green_radius, params.e_radius); // input radius

	// base address for thread block
	sx = blockIdx.x * 2 * blockDim.x + maxrad; // maxrad is apparently subject to be variable
	sy = blockIdx.y * 2 * blockDim.y + maxrad;

	// determine alignment of base address
	bankblk = get_gmem_alignment_diff(sx * sizeof(float), major, minor);
	bankblk /= sizeof(float);

	// determine total load length
	if (bankblk == 0)
		loopx = (int)ceil(((float)(2 * blockDim.x + params.window_radius))/(float)blockDim.x); // threads are alternatingly load each element
	else
		loopx = 1 + (int)ceil(((float)(2 * blockDim.x + params.window_radius))/(float)blockDim.x);

	loopy = (int)ceil(((float)(blockDim.y + params.window_radius / 2))/(float)blockDim.y);

	for (int y = 0; y < loopy; y++)
	{
		for (int x = 0; x < loopx; x++)
		{
			nsx = sx + x * blockDim.x + threadIdx.x - bankblk;
			nsy = sy + 2 * (y * blockDim.y + threadIdx.y);

			endx = sx + 2 * blockDim.x + params.window_radius;
			endy = sy + 2 * blockDim.y + params.window_radius;

			cx = nsx - maxrad;
			cy = nsy - maxrad;

			color = fc(filter, cx, cy);

			// adjust only y
			if (channel == COLOR_RED)
			{
				if (color == COLOR_GREEN_BLUE || color == COLOR_BLUE)
				{
					nsy--;
					cy--;
				}

			} else if (channel == COLOR_BLUE)
			{
				if (color == COLOR_GREEN_RED || color == COLOR_RED)
				{
					nsy++;
					cy++;
				}
			}

			// we need to check color of the thread once more
			color = fc(filter, cx, cy);

			if (color == channel)
			{
				if (nsx < endx && nsy < endy && nsx >= maxrad && nsy >= maxrad && 
					nsx < (params.width + maxrad + params.window_radius) && nsy < (params.height + maxrad + params.window_radius) &&
					nsx >= sx && nsy >= sy)
				{
					tx = (x * blockDim.x + threadIdx.x - bankblk) / 2;
					ty = y * blockDim.y + threadIdx.y;

					tgt[ty * (tgtptc / sizeof(float)) + tx] = input[nsy * (inputptc / sizeof(float)) + nsx];
				}
			}
		}
	}
}

// a function for testing vcd_sf_mp_load_input_ch. It copies a channel in shared memory to an output channel.
__device__ void vcd_sf_mp_test_load_input_ch_pos_k(float* out, size_t outptc, float* input, size_t inputptc, int channel, int filter, vcd_params params)
{
	int sx, sy, tx, ty, color = 0;

	sx = threadIdx.x;
	sy = threadIdx.y;

	tx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
	ty = 2 * (blockIdx.y * blockDim.y + threadIdx.y);

	// adjust position
	color = fc(filter, tx, ty);

	if (channel == COLOR_RED)
	{
		if (color == COLOR_GREEN_BLUE || color == COLOR_BLUE)
		{
			ty--;
		}

		if (color == COLOR_GREEN_RED || color == COLOR_RED)
		{
			tx--;
		}

	} else if (channel == COLOR_BLUE)
	{
		if (color == COLOR_GREEN_RED || color == COLOR_RED)
		{
			ty++;
		}

		if (color == COLOR_GREEN_BLUE || color == COLOR_RED)
		{
			tx++;
		}
	}

	if (tx < params.width && ty < params.height)
		out[ty * (outptc / sizeof(float)) + tx] = input[sy * (inputptc / sizeof(float)) + sx];
}

__device__ void vcd_sf_mp_calculate_variance_pos_k(float* out, size_t outptc, const float* ch, size_t chptc, 
												   const float* tempg, size_t tempgptc,int channel, int direction, int filter, vcd_params params)
{
	float res, res0, res2, res4 = 0.0f;
	int sx, sy, color = 0;

	sx = threadIdx.x;
	sy = threadIdx.y;

	if (direction == HORIZONTAL)
	{
		// calculate individual g in even position
		res0 = ch[sy * (chptc / sizeof(float)) + sx] - tempg[sy * (tempgptc / sizeof(float)) + sx];
		res2 = ch[sy * (chptc / sizeof(float)) + (sx + 1)] - tempg[sy * (tempgptc / sizeof(float)) + (sx + 1)];
		res4 = ch[sy * (chptc / sizeof(float)) + (sx + 2)] - tempg[sy * (tempgptc / sizeof(float)) + (sx + 2)];
		
	} else if (direction == VERTICAL)
	{
		// calculate individual g in even position
		res0 = ch[sy * (chptc / sizeof(float)) + sx] - tempg[sy * (tempgptc / sizeof(float)) + sx];
		res2 = ch[(sy + 1) * (chptc / sizeof(float)) + sx] - tempg[(sy + 1) * (tempgptc / sizeof(float)) + sx];
		res4 = ch[(sy + 2) * (chptc / sizeof(float)) + sx] - tempg[(sy + 2) * (tempgptc / sizeof(float)) + sx];
	}

	// calculate total
	res = res0 + res2 + res4 + (res0 + res2) / 2.0f + (res2 + res4) / 2.0f;

	// store position
	sx = blockIdx.x * blockDim.x + threadIdx.x;
	sy = 2 * (blockIdx.y * blockDim.y + threadIdx.y);

	color = fc(filter, sx, sy);

	// adjust only y
	if (channel == COLOR_RED)
	{
		if (color == COLOR_GREEN_BLUE || color == COLOR_BLUE)
		{
			sy--;
		}

	} else if (channel == COLOR_BLUE)
	{
		if (color == COLOR_GREEN_RED || color == COLOR_RED)
		{
			sy++;
		}
	}				

	// store result
	if (sy < params.height && sx < params.width / 2)
		out[sy * (outptc / sizeof(float)) + sx] = 8191;
}

__global__ void interpolate_g_dg_pos_k(
	float* out, size_t outptc,
	float* tempvarh, float* tempvarv, float* tempvardh, float* tempvardv, size_t tempvarptc,
	const float* tempgh, const float* tempgv, const float* tempgd, size_t tempgptc,
	const float* input, size_t inputptc,
	const int* e, size_t eptc,
	int filter, vcd_params params, smeminfo_interpolate_g_pos smeminfo,
	int major, int minor)
{
	// use sx register for calculating memory size
	int sx = (blockDim.y + params.window_radius / 2) * smeminfo.inputptcs / sizeof(float);
	
	extern __shared__ float ch_s[];
	float* tempg_s = &ch_s[sx];

	// process red channel

	// load input to separate channel
	vcd_sf_mp_load_input_ch_pos_k(ch_s, smeminfo.inputptcs, input, inputptc, COLOR_RED, filter, params, major, minor);
	
	/*vcd_sf_mp_test_load_input_ch_pos_k(out, outptc, ch_s, smeminfo.inputptcs, COLOR_RED, filter, params);
	__syncthreads();*/

	// load temporary green channels, and calculate variance
	vcd_sf_mp_load_tempg_k(tempg_s, smeminfo.tempgptcs, tempgh, tempgptc, blockIdx.x, blockIdx.y, COLOR_RED, filter, params);
	__syncthreads();
	vcd_sf_mp_calculate_variance_pos_k(tempvarh, tempvarptc, ch_s, smeminfo.inputptcs, tempg_s, smeminfo.tempgptcs, COLOR_RED, HORIZONTAL, filter, params);
	__syncthreads();

	vcd_sf_mp_load_tempg_k(tempg_s, smeminfo.tempgptcs, tempgv, tempgptc, blockIdx.x, blockIdx.y, COLOR_RED, filter, params);
	__syncthreads();
	vcd_sf_mp_calculate_variance_pos_k(tempvarv, tempvarptc, ch_s, smeminfo.inputptcs, tempg_s, smeminfo.tempgptcs, COLOR_RED, VERTICAL, filter, params);
	__syncthreads();

	vcd_sf_mp_load_tempg_k(tempg_s, smeminfo.tempgptcs, tempgd, tempgptc, blockIdx.x, blockIdx.y, COLOR_RED, filter, params);
	__syncthreads();
	vcd_sf_mp_calculate_variance_pos_k(tempvardh, tempvarptc, ch_s, smeminfo.inputptcs, tempg_s, smeminfo.tempgptcs, COLOR_RED, HORIZONTAL, filter, params);
	//__syncthreads();
	vcd_sf_mp_calculate_variance_pos_k(tempvardv, tempvarptc, ch_s, smeminfo.inputptcs, tempg_s, smeminfo.tempgptcs, COLOR_RED, VERTICAL, filter, params);
	__syncthreads();

	// process blue channel
	
	// load input to separate channel
	vcd_sf_mp_load_input_ch_pos_k(ch_s, smeminfo.inputptcs, input, inputptc, COLOR_BLUE, filter, params, major, minor);
	
	vcd_sf_mp_load_tempg_k(tempg_s, smeminfo.tempgptcs, tempgh, tempgptc, blockIdx.x, blockIdx.y, COLOR_BLUE, filter, params);
	__syncthreads();
	vcd_sf_mp_calculate_variance_pos_k(tempvarh, tempvarptc, ch_s, smeminfo.inputptcs, tempg_s, smeminfo.tempgptcs, COLOR_BLUE, HORIZONTAL, filter, params);
	__syncthreads();
	
	vcd_sf_mp_load_tempg_k(tempg_s, smeminfo.tempgptcs, tempgv, tempgptc, blockIdx.x, blockIdx.y, COLOR_BLUE, filter, params);
	__syncthreads();
	vcd_sf_mp_calculate_variance_pos_k(tempvarv, tempvarptc, ch_s, smeminfo.inputptcs, tempg_s, smeminfo.tempgptcs, COLOR_BLUE, VERTICAL, filter, params);
	__syncthreads();
	
	vcd_sf_mp_load_tempg_k(tempg_s, smeminfo.tempgptcs, tempgd, tempgptc, blockIdx.x, blockIdx.y, COLOR_BLUE, filter, params);
	__syncthreads();
	vcd_sf_mp_calculate_variance_pos_k(tempvardh, tempvarptc, ch_s, smeminfo.inputptcs, tempg_s, smeminfo.tempgptcs, COLOR_BLUE, HORIZONTAL, filter, params);
	//__syncthreads();
	vcd_sf_mp_calculate_variance_pos_k(tempvardv, tempvarptc, ch_s, smeminfo.inputptcs, tempg_s, smeminfo.tempgptcs, COLOR_BLUE, VERTICAL, filter, params);
	__syncthreads();

}

///////////////////////////////////////// VERSION 2: per-channel input ///////////////////////////////////////// 

void interpolate_g_dg_pos(float* out, size_t outptc,
	float* tempvarh, float* tempvarv, float* tempvarbh, float* tempvarbv, size_t tempvarptc,
	const float* tempgh, const float* tempgv, const float* tempgd, size_t tempgptc,
	const float* rr, const float* bb, size_t inputptc,
	const int* e, size_t eptc,
	int filter, vcd_params params, cudaDeviceProp prop, vcd_sf_mp_kernel_times* times)
{
	dim3 dimBlock, dimGrid;
	smeminfo_interpolate_g_pos info;

	float kerntime = 0.0f;
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	if (prop.major == 2)
		dimBlock.x = 32;
	else
		dimBlock.x = 16;
	dimBlock.y = 16;

	// shared memory size
	size_t smemsz = 0;
	
	// init smeminfo
	smeminfo_interpolate_g_pos_init(&info);
	info.inputptcs = get_smem_pitch((dimBlock.x + params.window_radius / 2) * sizeof(float)); // pitch for R channel and B channel
	info.tempgptcs = get_smem_pitch((dimBlock.x + params.window_radius / 2) * sizeof(float)); // pitch for temp G channel 
	
	smemsz += (info.inputptcs * (dimBlock.y + params.window_radius / 2)); // allocate for R and B channel
	smemsz += (3 * (info.tempgptcs * (dimBlock.y + params.window_radius / 2))); // allocate for temp G channel

	dimGrid.x = (int)ceil((float)(params.width / 2) / (float)dimBlock.x);
	dimGrid.y = (int)ceil((float)params.height / (float)dimBlock.y);

	cudaEventRecord(start, 0);
	interpolate_g_dg_pos_k<<<dimGrid, dimBlock, smemsz>>>(out, outptc, tempvarh, tempvarv, tempvarbh, tempvarbv, tempvarptc,
			tempgh, tempgv, tempgd, tempgptc, rr, bb, inputptc, e, eptc, filter, params, info, prop.major, prop.minor);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&kerntime, start, stop);

	times->calc_temp_var_pos = kerntime;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

}

// load input per channel in positive direction.
// version 2: does not use channel parameter, assume input is separated
__device__ void vcd_sf_mp_load_input_ch_pos_k(float* tgt, size_t tgtptc, const float* input, size_t inputptc, int filter, vcd_params params, int major, int minor)
{
	int tx, ty,loopx, loopy, nsx, nsy, color, bankblk, endx, endy, cx, cy = 0;
	int maxrad = max(params.window_radius + params.temp_green_radius, params.e_radius); // input radius

	// base address for thread block
	int sx = blockIdx.x * blockDim.x + maxrad / 2; // maxrad is apparently subject to be variable
	int sy = blockIdx.y * blockDim.y + maxrad / 2;

	// determine alignment of base address
	bankblk = get_gmem_alignment_diff(sx * sizeof(float), major, minor);
	bankblk /= sizeof(float);

	// determine total load length
	if (bankblk == 0)
		loopx = (int)ceil(((float)(blockDim.x + params.window_radius / 2))/(float)blockDim.x); // threads are alternatingly load each element
	else
		loopx = 1 + (int)ceil(((float)(blockDim.x + params.window_radius / 2))/(float)blockDim.x);

	loopy = (int)ceil(((float)(blockDim.y + params.window_radius / 2))/(float)blockDim.y);

	for (int y = 0; y < loopy; y++)
	{
		for (int x = 0; x < loopx; x++)
		{
			nsx = sx + x * blockDim.x + threadIdx.x - bankblk;
			nsy = sy + y * blockDim.y + threadIdx.y;

			endx = sx + blockDim.x + params.window_radius / 2;
			endy = sy + blockDim.y + params.window_radius / 2;
			
			if (nsx < endx && nsy < endy && nsx >= (maxrad / 2) && nsy >= (maxrad / 2) && 
				nsx < (params.width + maxrad + params.window_radius) / 2 && nsy < (params.height + maxrad + params.window_radius) / 2 &&
				nsx >= sx && nsy >= sy)
			{
				tx = (x * blockDim.x + threadIdx.x - bankblk);
				ty = y * blockDim.y + threadIdx.y;

				tgt[ty * (tgtptc / sizeof(float)) + tx] = input[nsy * (inputptc / sizeof(float)) + nsx];
			}
		
		}
	}
}

//////////////////////////////// VERSION 2 /////////////////////////////////////

__global__ void interpolate_g_dg_pos_k(
	float* out, size_t outptc,
	float* tempvarh, float* tempvarv, float* tempvardh, float* tempvardv, size_t tempvarptc,
	const float* tempgh, const float* tempgv, const float* tempgd, size_t tempgptc,
	const float* rr, const float* bb, size_t inputptc,
	const int* e, size_t eptc,
	int filter, vcd_params params, smeminfo_interpolate_g_pos smeminfo,
	int major, int minor)
{
	// use sx register for calculating memory size
	int sx = (blockDim.y + params.window_radius / 2) * smeminfo.inputptcs / sizeof(float);
	
	extern __shared__ float ch_s[];
	float* tempgh_s = &ch_s[sx];
	float* tempgv_s = &tempgh_s[sx];
	float* tempgd_s = &tempgv_s[sx];
	
	// process red channel

	// load input to separate channel
	vcd_sf_mp_load_input_ch_pos_k(ch_s, smeminfo.inputptcs, rr, inputptc, filter, params, major, minor);
	__syncthreads();
	
	// load temporary green channels
	vcd_sf_mp_load_tempg_k(tempgh_s, smeminfo.tempgptcs, tempgh, tempgptc, blockIdx.x, blockIdx.y, COLOR_RED, filter, params);
	__syncthreads();
	vcd_sf_mp_load_tempg_k(tempgv_s, smeminfo.tempgptcs, tempgv, tempgptc, blockIdx.x, blockIdx.y, COLOR_RED, filter, params);
	__syncthreads();
	vcd_sf_mp_load_tempg_k(tempgd_s, smeminfo.tempgptcs, tempgd, tempgptc, blockIdx.x, blockIdx.y, COLOR_RED, filter, params);
	__syncthreads();

	// calculate variance element in positive direction
	vcd_sf_mp_calculate_variance_pos_k(tempvarh, tempvarptc, ch_s, smeminfo.inputptcs, tempgh_s, smeminfo.tempgptcs, COLOR_RED, HORIZONTAL, filter, params);
	__syncthreads();
	vcd_sf_mp_calculate_variance_pos_k(tempvarv, tempvarptc, ch_s, smeminfo.inputptcs, tempgv_s, smeminfo.tempgptcs, COLOR_RED, VERTICAL, filter, params);
	__syncthreads();
	vcd_sf_mp_calculate_variance_pos_k(tempvardh, tempvarptc, ch_s, smeminfo.inputptcs, tempgd_s, smeminfo.tempgptcs, COLOR_RED, HORIZONTAL, filter, params);
	__syncthreads();
	vcd_sf_mp_calculate_variance_pos_k(tempvardv, tempvarptc, ch_s, smeminfo.inputptcs, tempgd_s, smeminfo.tempgptcs, COLOR_RED, VERTICAL, filter, params);
	__syncthreads();

	// process blue channel
	
	// load input to separate channel
	vcd_sf_mp_load_input_ch_pos_k(ch_s, smeminfo.inputptcs, bb, inputptc, filter, params, major, minor);
	__syncthreads();

	// load temporary green channels
	vcd_sf_mp_load_tempg_k(tempgh_s, smeminfo.tempgptcs, tempgh, tempgptc, blockIdx.x, blockIdx.y, COLOR_BLUE, filter, params);
	__syncthreads();
	vcd_sf_mp_load_tempg_k(tempgv_s, smeminfo.tempgptcs, tempgv, tempgptc, blockIdx.x, blockIdx.y, COLOR_BLUE, filter, params);
	__syncthreads();
	vcd_sf_mp_load_tempg_k(tempgd_s, smeminfo.tempgptcs, tempgd, tempgptc, blockIdx.x, blockIdx.y, COLOR_BLUE, filter, params);
	__syncthreads();

	// calculate variance element in positive direction
	vcd_sf_mp_calculate_variance_pos_k(tempvarh, tempvarptc, ch_s, smeminfo.inputptcs, tempgh_s, smeminfo.tempgptcs, COLOR_BLUE, HORIZONTAL, filter, params);
	__syncthreads();
	vcd_sf_mp_calculate_variance_pos_k(tempvarv, tempvarptc, ch_s, smeminfo.inputptcs, tempgv_s, smeminfo.tempgptcs, COLOR_BLUE, VERTICAL, filter, params);
	__syncthreads();
	vcd_sf_mp_calculate_variance_pos_k(tempvardh, tempvarptc, ch_s, smeminfo.inputptcs, tempgd_s, smeminfo.tempgptcs, COLOR_BLUE, HORIZONTAL, filter, params);
	__syncthreads();
	vcd_sf_mp_calculate_variance_pos_k(tempvardv, tempvarptc, ch_s, smeminfo.inputptcs, tempgd_s, smeminfo.tempgptcs, COLOR_BLUE, VERTICAL, filter, params);
	__syncthreads();
}