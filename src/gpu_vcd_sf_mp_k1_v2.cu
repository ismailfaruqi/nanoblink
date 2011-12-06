#include "gpu_vcd_sf_mp_k1_v2.cuh"
#include "gpu_common.cuh"
#include "gpu_vcd_common.cuh"
#include "vcd_common.h"

void stream_g_calculation(
	float* rg,  
	float* bg,
	size_t outptc,
	float* tempvarh, float* tempvarv, float* tempvarbh, float* tempvarbv, size_t tempvarptc,
	const float* tempgh, const float* tempgv, const float* tempgd, size_t tempgptc,
	const float* rr, const float* bb, size_t inputptc, 
	const int* e, size_t eptc,
	int filter, vcd_params params,
	cudaDeviceProp prop, vcd_sf_mp_kernel_times* times)
{
	float timer, timeb = 0.0f;
	const int streamNum = 1;
	cudaStream_t streams[streamNum];

	// create stream
	for (int i = 0; i < streamNum; i++)
		cudaStreamCreate(&streams[i]);

	// stream[0]: calculate G on R pixel, diagonal
	interpolate_g_dg_neg2(rg, outptc, tempvarh, tempvarv, tempvarbh, tempvarbv, tempvarptc, 
		tempgh, tempgv, tempgd, tempgptc, rr, inputptc, e, eptc, COLOR_RED, filter, params, prop, 0, &timer);

	// stream[1]: calculate G on B pixel, diagonal
	interpolate_g_dg_neg2(bg, outptc, tempvarh, tempvarv, tempvarbh, tempvarbv, tempvarptc, 
		tempgh, tempgv, tempgd, tempgptc, bb, inputptc, e, eptc, COLOR_BLUE, filter, params, prop, streams[0], &timeb);

	// destroy stream
	for (int i = 0; i < streamNum; i++)
		cudaStreamDestroy(streams[i]);
	
	// add time
	times->calc_var_neg = timer + timeb;
}

void interpolate_g_dg_neg2(float* out, size_t outptc,
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
	//smemsz += (4 * info.varptcs * dimBlock.y);
		
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
		interpolate_g_dg_neg_k2<<<dimGrid, dimBlock, smemsz, stream>>>(out, outptc, tempvarh, tempvarv, tempvarbh, tempvarbv, tempvarptc,
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
		interpolate_g_dg_neg_k2<<<dimGrid, dimBlock, smemsz, stream>>>(out, outptc, tempvarh, tempvarv, tempvarbh, tempvarbv, tempvarptc,
			tempgh, tempgv, tempgd, tempgptc, input, inputptc, e, eptc, channel, RIGHT_SWEEP, x, filter, params, info, prop.major, prop.minor);
		cudaEventRecord(stop,stream);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&kerntime,start,stop);
		*time += kerntime;
	}

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

// load temp green per channel
__device__ void vcd_sf_mp_load_tempg_k2(float* tgt, size_t tgtptc, const float* tempg, size_t tempgptc, 
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

// load input per channel in positive direction
__device__ void vcd_sf_mp_load_input_ch_neg_k2(float* tgt, size_t tgtptc, const float* input, size_t inputptc, int filter, vcd_params params, 
											  int bidx, int bidy, int major, int minor)
{
	int sx, sy, tx, ty,loopx, loopy, nsx, nsy, color, bankblk, endx, endy, cx, cy = 0;
	int maxrad = max(params.window_radius + params.temp_green_radius, params.e_radius); // input radius

	// base address for thread block
	sx = bidx * blockDim.x + params.temp_green_radius / 2;
	sy = bidy * blockDim.y + params.temp_green_radius / 2;

	// determine alignment of base address
	bankblk = get_gmem_alignment_diff(sx * sizeof(float), major, minor);
	bankblk /= sizeof(float);

	// determine total load length
	if (bankblk == 0)
		loopx = (int)ceil(((float)(blockDim.x + params.window_radius))/(float)blockDim.x); // threads are alternatingly load each element
	else
		loopx = 1 + (int)ceil(((float)(blockDim.x + params.window_radius))/(float)blockDim.x);

	loopy = (int)ceil(((float)(blockDim.y + params.window_radius))/(float)blockDim.y);

	for (int y = 0; y < loopy; y++)
	{
		for (int x = 0; x < loopx; x++)
		{
			nsx = sx + x * blockDim.x + threadIdx.x - bankblk;
			nsy = sy + y * blockDim.y + threadIdx.y;

			endx = sx + blockDim.x + params.window_radius;
			endy = sy + blockDim.y + params.window_radius;
			
			if (nsx < endx && nsy < endy && nsx >= (params.temp_green_radius / 2) && nsy >= (params.temp_green_radius / 2) && 
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

// load input per channel in positive direction
__device__ void vcd_sf_mp_load_output_neg_k2(float* tgt, size_t tgtptc, const float* src, size_t srcptc, int filter, vcd_params params, 
											  int bidx, int bidy, int major, int minor)
{
	int sx, sy, tx, ty,loopx, loopy, nsx, nsy, color, bankblk, endx, endy, cx, cy = 0;
	int maxrad = max(params.window_radius + params.temp_green_radius, params.e_radius); // input radius

	// base address for thread block
	sx = bidx * blockDim.x + params.temp_green_radius / 2;
	sy = bidy * blockDim.y + params.temp_green_radius / 2;

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
			
			if (nsx < endx && nsy < endy && nsx >= (params.temp_green_radius / 2) && nsy >= (params.temp_green_radius / 2) && 
				nsx < (params.width + maxrad + params.window_radius) / 2 && nsy < (params.height + maxrad + params.window_radius) / 2 &&
				nsx >= sx && nsy >= sy)
			{
				tx = (x * blockDim.x + threadIdx.x - bankblk);
				ty = y * blockDim.y + threadIdx.y;

				tgt[ty * (tgtptc / sizeof(float)) + tx] = src[nsy * (srcptc / sizeof(float)) + nsx];
			}
		
		}
	}
}

__device__ void vcd_sf_mp_store_outg_k2(float* outg, size_t outgptc, const float* outputg_s, size_t outputgptcs, int channel, int filter, vcd_params params, int bidx, int bidy)
{
	int sx, sy, tx, ty, ntx, color = 0;
	int maxrad = vcd_get_maxrad_k(params);

	tx = bidx * blockDim.x + threadIdx.x + maxrad / 2;
	ty = bidy * blockDim.y + threadIdx.y + maxrad / 2;

	sx = threadIdx.x + params.window_radius / 2;
	sy = threadIdx.y + params.window_radius / 2;

	if (ntx < (params.width + maxrad) / 2 && ty < (params.height + maxrad) / 2)
		outg[ty * (outgptc / sizeof(float)) + tx] = outputg_s[sy * (outputgptcs / sizeof(float)) + sx];
		
}

// calculate variance in vertical OR horizontal
__device__ void vcd_sf_mp_calculate_var_k2(float* var, float* inputch_s, size_t inputptcs, float* outputg_s, size_t outputgptcs, 
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
		float res2 = inputch_s[sy * (inputptcs / sizeof(float)) + (sx + 1)] - tempg_s[tdy * (outputgptcs / sizeof(float)) + (tdx + 1)];
		float res4 = inputch_s[sy * (inputptcs / sizeof(float)) + (sx + 2)] - tempg_s[tdy * (outputgptcs / sizeof(float)) + (tdx + 2)];

		float res = res0 + res2 + res4 + resm2 + resm4 + (res0 + resm2) / 2.0f + (resm2 + resm4) / 2.0f + (res0 + res2) / 2.0f + (res2 + res4) / 2.0f; // sigma
		res /= 9.0f; // 1/9 sigma

		// each of (d - sigma)^2
		res0 -= res;
		res0 *= res0;
		res2 -= res;
		res2 *= res2;
		res4 -= res;
		res4 *= res4;
		resm2 -= res;
		resm2 *= resm2;
		resm4 -= res;
		resm4 *= resm4;

		// 1/9 * sigma (d - sigma)^2
		*var = (res0 + res2 + res4 + resm2 + resm4 + (res0 + res2) / 2.0f + (res2 + res4) / 2.0f + (res0 + resm2) / 2.0f + (resm2 + resm4) / 2.0f) / 9.0f;

	} else if (direction == VERTICAL)
	{
		float res0 = inputch_s[sy * (inputptcs / sizeof(float)) + sx] - tempg_s[tdy * (tempgptcs / sizeof(float)) + tdx];
		float resm2 = inputch_s[(sy - 1) * (inputptcs / sizeof(float)) + sx] - outputg_s[(sy - 1)* (outputgptcs / sizeof(float)) + sx];
		float resm4 = inputch_s[(sy - 2) * (inputptcs / sizeof(float)) + sx] - outputg_s[(sy - 2) * (outputgptcs / sizeof(float)) + sx];
		float res2 = inputch_s[(sy + 1) * (inputptcs / sizeof(float)) + sx] - tempg_s[(tdy + 1) * (outputgptcs / sizeof(float)) + tdx];
		float res4 = inputch_s[(sy + 2) * (inputptcs / sizeof(float)) + sx] - tempg_s[(tdy + 2) * (outputgptcs / sizeof(float)) + tdx];

		float res = res0 + res2 + res4 + resm2 + resm4 + (res0 + resm2) / 2.0f + (resm2 + resm4) / 2.0f + (res0 + res2) / 2.0f + (res2 + res4) / 2.0f; // sigma
		res /= 9.0f; // 1/9 sigma

		// each of (d - sigma)^2
		res0 -= res;
		res0 *= res0;
		res2 -= res;
		res2 *= res2;
		res4 -= res;
		res4 *= res4;
		resm2 -= res;
		resm2 *= resm2;
		resm4 -= res;
		resm4 *= resm4;

		// 1/9 * sigma (d - sigma)^2
		*var = (res0 + res2 + res4 + resm2 + resm4 + (res0 + res2) / 2.0f + (res2 + res4) / 2.0f + (res0 + resm2) / 2.0f + (resm2 + resm4) / 2.0f) / 9.0f;

	}

}

// calculate variance in diagonal 
__device__ void vcd_sf_mp_calculate_vard_k2(float* var, float* inputch_s, size_t inputptcs, float* outputg_s, size_t outputgptcs, 
				const float* tempg_s, size_t tempgptcs, 
				int tdx, int tdy, int window_radius)
{
	int sx = tdx + window_radius / 2;
	int sy = tdy + window_radius / 2;

	// horizontal element
	float res0 = inputch_s[sy * (inputptcs / sizeof(float)) + sx] - tempg_s[tdy * (tempgptcs / sizeof(float)) + tdx];
	float resm2 = inputch_s[sy * (inputptcs / sizeof(float)) + (sx - 1)] - outputg_s[sy * (outputgptcs / sizeof(float)) + (sx - 1)];
	float resm4 = inputch_s[sy * (inputptcs / sizeof(float)) + (sx - 2)] - outputg_s[sy * (outputgptcs / sizeof(float)) + (sx - 2)];
	float res2 = inputch_s[sy * (inputptcs / sizeof(float)) + (sx + 1)] - tempg_s[tdy * (outputgptcs / sizeof(float)) + (tdx + 1)];
	float res4 = inputch_s[sy * (inputptcs / sizeof(float)) + (sx + 2)] - tempg_s[tdy * (outputgptcs / sizeof(float)) + (tdx + 2)];

	float resh = res0 + res2 + res4 + resm2 + resm4 + (res0 + resm2) / 2.0f + (resm2 + resm4) / 2.0f + (res0 + res2) / 2.0f + (res2 + res4) / 2.0f; // sigma
	resh /= 9.0f; // 1/9 sigma

	// each of (d - sigma)^2
	res0 -= resh;
	res0 *= res0;
	res2 -= resh;
	res2 *= res2;
	res4 -= resh;
	res4 *= res4;
	resm2 -= resh;
	resm2 *= resm2;
	resm4 -= resh;
	resm4 *= resm4;

	// 1/9 * sigma (d - sigma)^2
	resh = (res0 + res2 + res4 + resm2 + resm4 + (res0 + res2) / 2.0f + (res2 + res4) / 2.0f + (res0 + resm2) / 2.0f + (resm2 + resm4) / 2.0f) / 9.0f;

	// vertical element
	res0 = inputch_s[sy * (inputptcs / sizeof(float)) + sx] - tempg_s[tdy * (tempgptcs / sizeof(float)) + tdx];
	resm2 = inputch_s[(sy - 1) * (inputptcs / sizeof(float)) + sx] - outputg_s[(sy - 1)* (outputgptcs / sizeof(float)) + sx];
	resm4 = inputch_s[(sy - 2) * (inputptcs / sizeof(float)) + sx] - outputg_s[(sy - 2) * (outputgptcs / sizeof(float)) + sx];
	res2 = inputch_s[(sy + 1) * (inputptcs / sizeof(float)) + sx] - tempg_s[(tdy + 1) * (outputgptcs / sizeof(float)) + tdx];
	res4 = inputch_s[(sy + 2) * (inputptcs / sizeof(float)) + sx] - tempg_s[(tdy + 2) * (outputgptcs / sizeof(float)) + tdx];

	float resv = res0 + res2 + res4 + resm2 + resm4 + (res0 + resm2) / 2.0f + (resm2 + resm4) / 2.0f + (res0 + res2) / 2.0f + (res2 + res4) / 2.0f; // sigma
	resv /= 9.0f; // 1/9 sigma

	// each of (d - sigma)^2
	res0 -= resv;
	res0 *= res0;
	res2 -= resv;
	res2 *= res2;
	res4 -= resv;
	res4 *= res4;
	resm2 -= resv;
	resm2 *= resm2;
	resm4 -= resv;
	resm4 *= resm4;

	// 1/9 * sigma (d - sigma)^2
	resv = (res0 + res2 + res4 + resm2 + resm4 + (res0 + res2) / 2.0f + (res2 + res4) / 2.0f + (res0 + resm2) / 2.0f + (resm2 + resm4) / 2.0f) / 9.0f;

	// write output
	*var = (resh + resv) / 2.0f;
}

__global__ void interpolate_g_dg_neg_k2(
	float* out, size_t outptc,
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
	
	// load input
	vcd_sf_mp_load_input_ch_neg_k2(inputch_s, smeminfo.inputptcs, input, inputptc, filter, params, bdx, bdy, major, minor);
	__syncthreads();

	// load output. Pad them with zero in the left and top apron
	vcd_sf_mp_load_output_neg_k2(outputg_s, smeminfo.outputgptcs, out, outptc, filter, params, bdx, bdy, major, minor);
	__syncthreads();

	// load temporary greens, to avoid uncoalesced access on global memory
	vcd_sf_mp_load_tempg_k2(tempgh_s, smeminfo.tempgptcs, tempgh, tempgptc, bdx, bdy, channel, filter, params);
	vcd_sf_mp_load_tempg_k2(tempgv_s, smeminfo.tempgptcs, tempgv, tempgptc, bdx, bdy, channel, filter, params);
	vcd_sf_mp_load_tempg_k2(tempgd_s, smeminfo.tempgptcs, tempgd, tempgptc, bdx, bdy, channel, filter, params);
	__syncthreads();

	// calculate variance	

	// use only the first warp (fermi) or half warp ( < fermi) of the thread. Assume that blockDim.x >= blockDim.y
	
	// calculate variance element in down sweep direction
	for (int i = 0; i < blockDim.y; i++)
	{
		int tdx = threadIdx.x;
		int tdy = i - threadIdx.x;

		if (tdx >= 0 && tdx <= i && tdy >= 0 && tdy <= i && threadIdx.y == 0)
		{
			float varh, varv, vard = 0.0f; // loopx will hold horizontal calc. result. loopy will hold vertical calc. result

			vcd_sf_mp_calculate_var_k2(&varh, inputch_s, smeminfo.inputptcs, outputg_s, smeminfo.outputgptcs, 
					tempgh_s, smeminfo.tempgptcs, 
					HORIZONTAL, tdx, tdy, params.window_radius);

			vcd_sf_mp_calculate_var_k2(&varv, inputch_s, smeminfo.inputptcs, outputg_s, smeminfo.outputgptcs, 
					tempgv_s, smeminfo.tempgptcs, 
					VERTICAL, tdx, tdy, params.window_radius);

			vcd_sf_mp_calculate_vard_k2(&vard, inputch_s, smeminfo.inputptcs, outputg_s, smeminfo.outputgptcs, 
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

		int tdx = threadIdx.x + i;
		int tdy = blockDim.y - threadIdx.x - 1;

		if (tdx >= 0 && tdx < blockDim.x && tdy >= 0 && tdy < blockDim.y &&
			(tdx + tdy) == (blockDim.y + i - 1) && threadIdx.y == 0)
		{
			float varh, varv, vard = 0.0f; // loopx will hold horizontal calc. result. loopy will hold vertical calc. result

			vcd_sf_mp_calculate_var_k2(&varh, inputch_s, smeminfo.inputptcs, outputg_s, smeminfo.outputgptcs, 
					tempgh_s, smeminfo.tempgptcs, 
					HORIZONTAL, tdx, tdy, params.window_radius);

			vcd_sf_mp_calculate_var_k2(&varv, inputch_s, smeminfo.inputptcs, outputg_s, smeminfo.outputgptcs, 
					tempgv_s, smeminfo.tempgptcs, 
					VERTICAL, tdx, tdy, params.window_radius);

			vcd_sf_mp_calculate_vard_k2(&vard, inputch_s, smeminfo.inputptcs, outputg_s, smeminfo.outputgptcs, 
					tempgd_s, smeminfo.tempgptcs, 
					tdx, tdy, params.window_radius);
			
			vcd_sf_mp_determinte_g_k(outputg_s, smeminfo.outputgptcs,
				tempgh_s, tempgv_s, tempgd_s, smeminfo.tempgptcs, smeminfo.tempglen,
				varh, varv, vard, tdx, tdy, params.window_radius);

			
		}

		__syncthreads();
	}

	// store output G
	vcd_sf_mp_store_outg_k2(out, outptc, outputg_s, smeminfo.outputgptcs, channel, filter, params, bdx, bdy);
	
}