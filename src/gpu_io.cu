#include "gpu_io.cuh"
#include "gpu_common.cuh"

// get y correction given filter color target and source
__device__ int correct_y(vcd_params params, int tgtcolor, int thiscolor);

// get x correction given filter color target and source
__device__ int correct_x(vcd_params params, int tgtcolor, int thiscolor);

__global__ void convert_input_k(float* input_g, size_t input_g_pitch, 
	const unsigned short *input4_g, size_t input4Pitch,
	size_t smemPitch, int filter, vcd_params params)
{
	extern __shared__ float smem[];
	unsigned int* converted = (unsigned int*)input4_g;
	int maxrad = max(params.window_radius + params.temp_green_radius,params.e_radius);

	// step 1: load input to shared memory
	int sx,sy,tx,ty = 0;
	
	sx = blockDim.x * blockIdx.x + threadIdx.x;
	sy = blockDim.y * blockIdx.y + threadIdx.y;
	
	tx = threadIdx.x; // thread target address x
	ty = 2 * threadIdx.y; // thread target address y

	if (sx < ((params.width + 2 * maxrad) / 2) && sy < (params.height + 2 * maxrad))
	{
		unsigned int val = converted[sy * (input4Pitch / sizeof(unsigned int)) + sx];
		smem[ty * (smemPitch / sizeof(float)) + tx] = (float)(val & 0xFFFF);
		smem[(ty + 1) * (smemPitch / sizeof(float)) + tx] = (float)(val >> 16);			
	}

	__syncthreads();

	// step 2: store to global memory
	for (int i = 0; i < 2; i++)
	{
		sx = (i * blockDim.x + threadIdx.x) / 2; // thread source address x
		sy = 2 * threadIdx.y + (threadIdx.x % 2); // thread source address y

		tx = 2 * blockDim.x * blockIdx.x + i * blockDim.x + threadIdx.x;
		ty = blockDim.y * blockIdx.y + threadIdx.y;

		if (tx < (params.width + 2 * maxrad) && ty < (params.height + 2 * maxrad))
		{
			input_g[ty * (input_g_pitch / sizeof(float)) + tx] = smem[sy * (smemPitch / sizeof(float)) + sx];
		}
	}
}

void convert_input(float* input_g, size_t input_g_pitch,
				   const unsigned short* input4_g, size_t input4_pitch,
				   int filter, vcd_params params, cudaDeviceProp prop, vcd_sf_mp_kernel_times* times)
{
	dim3 dimBlock, dimGrid;
	size_t smemPitch, smemSize;
	int totalApronRadius = max((params.window_radius + params.temp_green_radius),params.e_radius);
	int banks = 0;

	float kerntime = 0.0f;
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	if (prop.major == 2)
	{
		dimBlock.x = 32;
		banks = 32;
	}
	else // prop.major = 1.x
	{
		dimBlock.x = 32;
		banks = 16;
	}

	dimBlock.y = 16;
	dimGrid.x = (int) ceil((float)(params.width + 2 * totalApronRadius) / (float)(2 * dimBlock.x));
	dimGrid.y = (int) ceil((float)(params.height + 2 * totalApronRadius) / (float)dimBlock.y);
	smemPitch = (dimBlock.x + banks / 2) * sizeof(float);
	smemSize = smemPitch * (dimBlock.y * 2);

	cudaEventRecord(start, 0);
	convert_input_k<<<dimGrid, dimBlock, smemSize>>>(input_g, input_g_pitch, input4_g, input4_pitch, smemPitch, filter, params);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&kerntime, start, stop);

	times->convert_input = kerntime;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

__global__ void convert_input_test(unsigned int* output_R, 
	unsigned int* output_G, unsigned int* output_B, size_t output_pitch, 
	float* input_g, size_t input_g_pitch, vcd_params params)
{
	int maxrad = max(params.window_radius + params.temp_green_radius,params.e_radius);

	int sx = blockIdx.x * blockDim.x + threadIdx.x + maxrad;
	int sy = blockIdx.y * blockDim.y + threadIdx.y + maxrad;

	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;

	if (tx < (params.width) && ty < (params.height))
	{
		unsigned int val = (unsigned int)input_g[sy * (input_g_pitch / sizeof(float)) + sx];
		output_R[ty * (output_pitch / sizeof(unsigned int)) + tx] = val;
		output_G[ty * (output_pitch / sizeof(unsigned int)) + tx] = val;
		output_B[ty * (output_pitch / sizeof(unsigned int)) + tx] = val;
	}
}

__global__ void convert_input_test2(unsigned short* output, size_t output_pitch, const float* input_g, size_t input_g_pitch, vcd_params params)
{
	int maxrad = max(params.window_radius + params.temp_green_radius,params.e_radius);

	int sx = blockIdx.x * blockDim.x + threadIdx.x + maxrad;
	int sy = blockIdx.y * blockDim.y + threadIdx.y + maxrad;

	int tx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
	int ty = blockIdx.y * blockDim.y + threadIdx.y;

	if (tx < (4 * params.width) && ty < (params.height))
	{
		unsigned short val = (unsigned short)input_g[sy * (input_g_pitch / sizeof(float)) + sx];
		output[ty * (output_pitch / sizeof(unsigned short)) + tx] = val;
		output[ty * (output_pitch / sizeof(unsigned short)) + tx + 1] = val;
		output[ty * (output_pitch / sizeof(unsigned short)) + tx + 2] = val;
		output[ty * (output_pitch / sizeof(unsigned short)) + tx + 3] = 8191;
	}
}

__global__ void convert_output_k(unsigned short* output, size_t output_pitch,
	const float* output_R, const float* output_G,
	const float* output_B, size_t output_sc_pitch, size_t smemPitch, vcd_params params)
{
	// each thread loads one int from each channel, therefore achieving full coalescing
	int sx = blockIdx.x * blockDim.x + threadIdx.x;
	int sy = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned int R, G, B = 0;
	
	if (sx < params.width && sy < params.height)
	{
		R = (unsigned int)output_R[sy * (output_sc_pitch / sizeof(unsigned int)) + sx];
		G = (unsigned int)output_G[sy * (output_sc_pitch / sizeof(unsigned int)) + sx];
		B = (unsigned int)output_B[sy * (output_sc_pitch / sizeof(unsigned int)) + sx];
	}

	int tx = 2 * threadIdx.x;
	int ty = threadIdx.y;
	extern __shared__ unsigned int smem2[];

	// store with bank conflict.
	smem2[ty * (smemPitch / sizeof(int)) + tx + 0] = ((G & 0xFFFF) << 16) | (B & 0xFFFF);
	smem2[ty * (smemPitch / sizeof(int)) + tx + 1] = (8191 << 16) | (R & 0xFFFF);
	
	// put values in SMEM into output in global memory as integer to achieve coalescing
	sx = 2 * threadIdx.x;
	sy = threadIdx.y;

	tx = 2 * blockIdx.x * blockDim.x;
	ty = blockIdx.y * blockDim.y + threadIdx.y;

	for (int i = 0; i < 2; i++)
	{
		int nx = tx + i * blockDim.x + threadIdx.x;
		if (nx < 2 * params.width && ty < params.height)
		{
			((int*)output)[ty * (output_pitch / sizeof(int)) + nx] = ((unsigned int*)smem2)[sy * (smemPitch / sizeof(unsigned int)) + blockDim.x * i + threadIdx.x];
		} 
	}
}



void convert_output(unsigned short* output, size_t output_pitch,
	const float* output_R, const float* output_G,
	const float* output_B, size_t output_sc_pitch, vcd_params params, cudaDeviceProp prop, vcd_sf_mp_kernel_times* times)
{
	dim3 dimBlock, dimGrid;
	size_t smemPitch, smemSize;

	if (prop.major == 2)
		dimBlock.x = 32;
	else // prop.major = 1.x
		dimBlock.x = 16;
	dimBlock.y = 16;

	float kerntime = 0.0f;
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dimGrid.x = (int) ceil((float)params.width / (float)dimBlock.x);
	dimGrid.y = (int) ceil((float)params.height / (float)dimBlock.y);
	smemPitch = get_smem_pitch(4 * dimBlock.x * sizeof(unsigned int));
	smemSize = smemPitch * dimBlock.y;

	cudaEventRecord(start, 0);
	convert_output_k<<<dimGrid, dimBlock, smemSize>>>(output, output_pitch, output_R, output_G, output_B, output_sc_pitch, smemPitch, params);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&kerntime, start, stop);

	times->convert_output = kerntime;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

}

__global__ void separate_input_k(float* rr, float* bb, float* grg, float* gbg, size_t chptc, 
								 const float* input, size_t inputptc, size_t inputptcs, int filter, vcd_params params)
{
	int maxrad = vcd_get_maxrad_k(params);

	// first, load input to shared memory
	// IMPORTANT: group the same channel into consecutive row and consecutive column
	extern __shared__ float input_s[];

	for (int x = 0; x < 2; x++)
	{
		int sx = 2 * blockIdx.x * blockDim.x + x * blockDim.x + threadIdx.x;
		int sy = blockIdx.y * blockDim.y + threadIdx.y;

		// take value
		if (sx < (params.width + 2 * maxrad) &&
			sy < (params.height + 2 * maxrad))
		{
			unsigned int val = input[sy * (inputptc / sizeof(unsigned int)) + sx];
			
			int tx = (x * blockDim.x + threadIdx.x) / 2;
			int ty = 2 * threadIdx.y + (threadIdx.x % 2);

			input_s[ty * (inputptcs / sizeof(float)) + tx] = val;
			
		}
	}	

	__syncthreads();

	// then put into each respective channel
	for (int y = 0; y < 2; y++)
	{
		int sx = threadIdx.x;
		int sy = 2 * threadIdx.y + y;

		int cx = 2 * blockIdx.x * blockDim.x + 2 * threadIdx.x + sy % 2;
		int cy = blockIdx.y * blockDim.y + threadIdx.y;

		int color = fc(filter, cx, cy);

		if (cx < (params.width + 2 * maxrad) &&
			cy < (params.height + 2 * maxrad))
		{

			int tx = blockIdx.x * blockDim.x + threadIdx.x;
			int ty = (blockIdx.y * blockDim.y + threadIdx.y) / 2;
			
			if (color == COLOR_RED)
			{
				rr[ty * (chptc / sizeof(float)) + tx] = input_s[sy * (inputptcs / sizeof(float)) + sx];
			} else if (color == COLOR_BLUE)
			{
				bb[ty * (chptc / sizeof(float)) + tx] = input_s[sy * (inputptcs / sizeof(float)) + sx];
			} else if (color == COLOR_GREEN_RED)
			{
				grg[ty * (chptc / sizeof(float)) + tx] = input_s[sy * (inputptcs / sizeof(float)) + sx];
			} else if (color == COLOR_GREEN_BLUE)
			{
				gbg[ty * (chptc / sizeof(float)) + tx] = input_s[sy * (inputptcs / sizeof(float)) + sx];
			}
		}
	}
	
}

void separate_input(float* rr, float* bb, float* grg, float* gbg, size_t chptc,
					const float* input, size_t inputptc, int filter, vcd_params params, cudaDeviceProp prop,
					vcd_sf_mp_kernel_times* times)
{
	dim3 dimBlock, dimGrid;
	size_t chptcs, inputptcs, smemSize = 0;
	int maxrad = vcd_get_maxrad(params);
	int banks = 0;

	if (prop.major == 2)
	{
		dimBlock.x = 32;
		banks = 32;
	}
	else // prop.major = 1.x
	{
		dimBlock.x = 16;
		banks = 16;
	}
	dimBlock.y = 16;
	
	dimGrid.x = (int)ceil((float)(params.width + 2 * maxrad) / (2.0f * dimBlock.x));
	dimGrid.y = (int)ceil((float)(params.height + 2 * maxrad) / (float)dimBlock.y);

	// shared memory size
	inputptcs = (dimBlock.x + (banks / 2)) * sizeof(float);
	
	smemSize += (inputptcs * 2 * dimBlock.y);

	float kerntime = 0.0f;
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// launch kernel
	separate_input_k<<<dimGrid, dimBlock, smemSize>>>(rr, bb, grg, gbg, chptc, input, inputptc, inputptcs, filter, params);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&kerntime, start, stop);

	times->separate_input = kerntime;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

// combine input kernel. Lets do lousy combining here: never mind have 2-way bank conflict.
__global__ void combine_channels_k(unsigned short* output, size_t outptc,
								 float *rr, float* rg, float* rb,
								 float *br, float* bg, float* bb,
								 float *grr, float* grg, float* grb,
								 float* gbr, float* gbg, float* gbb, size_t inchptc,
								 size_t stgptcs,
								 int filter,
								 vcd_params params)
{
	int maxrad = vcd_get_maxrad_k(params);

	extern __shared__ unsigned int temp_s[];

	for (int x = 0; x < 2; x++)
	{
		int cx = 2 * (blockIdx.x * blockDim.x + threadIdx.x) + x;
		int cy = blockIdx.y * blockDim.y + threadIdx.y;
	
		int color = fc(filter, cx, cy);

		if (cx < params.width && cy < params.height)
		{
			int nsy = (blockIdx.y * blockDim.y + threadIdx.y + maxrad) / 2;
			int nsx = blockIdx.x * blockDim.x + threadIdx.x + maxrad / 2;
			
			// load from each channel
			float *sr, *sg, float* sb;

			if (color == COLOR_RED)
			{
				sr = rr;
				sg = rg;
				sb = rb;
			} else if (color == COLOR_BLUE)
			{
				sr = br;
				sg = bg;
				sb = bb;
			} else if (color == COLOR_GREEN_RED)
			{
				sr = grr;
				sg = grg;
				sb = grb;
			} else if (color == COLOR_GREEN_BLUE)
			{
				sr = gbr;
				sg = gbg;
				sb = gbb;
			}

			unsigned int R = (unsigned int)sr[nsy * (inchptc / sizeof(float)) + nsx];
			unsigned int G = (unsigned int)sg[nsy * (inchptc / sizeof(float)) + nsx];
			unsigned int B = (unsigned int)sb[nsy * (inchptc / sizeof(float)) + nsx];

			// combine
			unsigned int BGx = ((G & 0xFFFF) << 16) | (B & 0xFFFF);
			unsigned int RAx = (8191 << 16) | (R & 0xFFFF);
			
			// put into shared memory
			int tx = threadIdx.x;
			int ty = 4 * threadIdx.y + 2 * x;

			temp_s[ty * (stgptcs / sizeof(unsigned int)) + tx] = BGx;
			temp_s[(ty + 1) * (stgptcs / sizeof(unsigned int)) + tx]  = RAx;
		}
	}

	__syncthreads();
	
	// put into global memory

	for (int i = 0; i < 4; i++)
	{
		int sx = (i * blockDim.x + threadIdx.x) / 4;
		int sy = 4 * threadIdx.y + threadIdx.x % 4;

		int tx = 4 * blockIdx.x * blockDim.x + i * blockDim.x + threadIdx.x;
		int ty = blockIdx.y * blockDim.y + threadIdx.y;

		if ((tx / 4) < params.width && ty < params.height)
			((unsigned int*)output)[ty * (outptc / sizeof(unsigned int)) + tx] = temp_s[sy * (stgptcs / sizeof(unsigned int)) + sx];
	}

	
}

void combine_channels(unsigned short* output, size_t outptc,
					  float *rr, float* rg, float* rb,
								 float *br, float* bg, float* bb,
								 float *grr, float* grg, float* grb,
								 float* gbr, float* gbg, float* gbb, size_t inchptc, int filter, vcd_params params, cudaDeviceProp prop,
								 vcd_sf_mp_kernel_times *times)
{
	dim3 dimGrid, dimBlock;
	size_t smemsz = 0; 
	size_t banks = 0;

	if (prop.major == 2)
	{
		dimBlock.x = 32;
		banks = 32;
	} else
	{
		dimBlock.x = 16;
		banks = 16;
	}
	dimBlock.y = 16;
	
	dimGrid.x = (int)ceil((float)params.width / (2.0f * dimBlock.x));
	dimGrid.y = (int)ceil((float)params.height / (float)dimBlock.y);
	
	// shared memory for temporarily staging
	size_t stgptcs = ((dimBlock.x + banks / 4) * sizeof(unsigned int));
	smemsz += (stgptcs * 4 * dimBlock.y);

	float kerntime = 0.0f;
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// launch kernel
	combine_channels_k<<<dimGrid, dimBlock, smemsz>>>(output, outptc, rr, rg, rb, br, bg, bb, grr, grg, grb, gbr, gbg, gbb, inchptc, stgptcs, filter, params);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&kerntime, start, stop);

	times->calc_rest = kerntime;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

