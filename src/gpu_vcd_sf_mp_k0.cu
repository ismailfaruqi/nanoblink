#include "gpu_common.cuh"
#include "gpu_vcd_common.cuh"
#include "gpu_vcd_sf_mp_k0.cuh"

__device__ float k0_get_input_g(float* input_g, size_t input_g_pitch, int x, int y)
{
	return *((float*)((char*)&input_g[0] + input_g_pitch * y) + x);
}

// set an element to green
__device__ void set_tempGreen(float* tempGreen_g, size_t tempGreen_g_pitch, int x, int y, float value)
{
	*((float*)((char*)&tempGreen_g[0] + tempGreen_g_pitch * y) + x) = value;
}

__device__ void set_e(int* eValues_g, size_t eValues_g_pitch, int x, int y, int value)
{
	*((int*)((char*)&eValues_g[0] + eValues_g_pitch * y) + x) = value;
}

// set an element to shared memory
__device__ void k0_set_input_s(float* input_s, size_t tempGreen_s_pitch, int x, int y, float val)
{
	input_s[y * (tempGreen_s_pitch / sizeof(float)) + x] = val;
}

__device__ void k0_set_input_s_i(int* input_s, size_t tempGreen_s_pitch, int x, int y, int val)
{
	input_s[y * (tempGreen_s_pitch / sizeof(float)) + x] = val;
}

// get an element from shared memory
__device__ float k0_get_input_s(float* input_s, size_t tempGreen_s_pitch, int x, int y)
{
	return input_s[y * (tempGreen_s_pitch / sizeof(float)) + x];
}

__device__ int k0_get_input_s_i(int* input_s, size_t tempGreen_s_pitch, int x, int y)
{
	return input_s[y * (tempGreen_s_pitch / sizeof(float)) + x];
}

__global__ void calculate_temp_green_k(float* input_g, size_t input_g_pitch, 
	float* tempGreenV_g, float* tempGreenH_g, float* tempGreenD_g,
	size_t tempGreen_g_pitch, size_t tempGreen_s_pitch, size_t staging_pitch, vcd_params params, int filter)
{
	extern __shared__ float input_s[];
	int loopx, loopy = 0;

	int maxtr = max(params.window_radius + params.temp_green_radius, params.e_radius); // apron size, max transfer radius
	
	int tlsx, tlsy, tltx, tlty = 0;
	float valV, valH, valD = 0.0f;

	// load input from global memory to shared memory
	tlsx = 2 * blockDim.x * blockIdx.x + params.window_radius;
	tlsy = blockDim.y * blockIdx.y + params.window_radius;

	int bankblk = tlsx % 16;

	loopy = (int)ceil((float)(blockDim.y + 2 * params.temp_green_radius)/(blockDim.y));
	if (bankblk == 0)
		loopx = (int)ceil((float)(2 * blockDim.x + 2 * params.temp_green_radius)/(blockDim.x));
	else
		loopx = 1 + (int)ceil((float)(2 * blockDim.x + 2 * params.temp_green_radius - (16 - bankblk))/(blockDim.x));

	for (int y = 0; y < loopy; y++)
	{
		for (int x = 0; x < loopx; x++)
		{
			int ntlsx = tlsx + x * blockDim.x + threadIdx.x - bankblk; // thread load source address, x
			int ntlsy = tlsy + y * blockDim.y + threadIdx.y; // thread load source address, y

			int endx = tlsx + 2 * blockDim.x + 2 * params.temp_green_radius;
			int endy = tlsy + blockDim.y + 2 * params.temp_green_radius;

			if (ntlsx < endx && ntlsy < endy && ntlsx >= tlsx && // load condition
				ntlsx < (params.width + 2 * maxtr) && ntlsy < (params.height + 2 * maxtr)) // border condition
			{
				tltx = x * blockDim.x + threadIdx.x - bankblk;
				tlty = y * blockDim.y + threadIdx.y;

				float elmt = k0_get_input_g(input_g, input_g_pitch, ntlsx, ntlsy); // get element from load source
				k0_set_input_s(input_s, tempGreen_s_pitch, tltx, tlty, elmt);

			}
		}
	}

	// make sure all elements are loaded
	__syncthreads();

	tlsx = 2 * (threadIdx.x % (blockDim.x / 2)) + (threadIdx.y / (blockDim.y / 2)) * blockDim.x + params.temp_green_radius; // x address for loading from input
	tlsy = threadIdx.x / (blockDim.x / 2) + 2 * (threadIdx.y % (blockDim.y / 2)) + params.temp_green_radius; // y address for loading from input

	tltx = params.x + 2 * blockDim.x * blockIdx.x + tlsx;
	tlty = params.y + blockDim.y * blockIdx.y + tlsy;

	int col = fc(filter, tltx, tlty);
	if (col == COLOR_GREEN_RED)
	{
		tlsx += params.grtor;
		tltx += params.grtor;
	}
	else if (col == COLOR_GREEN_BLUE)
	{
		tlsx += params.gbtob;
		tltx += params.gbtob;
	}
	
	// SHARED MEMORY VERSION
	// calculate vertical green
	if (tltx < (params.x + params.width + params.window_radius) &&
		tlty < (params.y + params.height + params.window_radius))
	{
		valV = (k0_get_input_s(input_s, tempGreen_s_pitch, tlsx, tlsy-1) + 
			k0_get_input_s(input_s, tempGreen_s_pitch, tlsx, tlsy+1)) / 2.0f + 
			(2.0f * k0_get_input_s(input_s, tempGreen_s_pitch, tlsx, tlsy) - 
			k0_get_input_s(input_s, tempGreen_s_pitch, tlsx, tlsy-2) - 
			k0_get_input_s(input_s, tempGreen_s_pitch, tlsx, tlsy+2)) / 4.0f;
	
		// calculate horizontal green
		valH = (k0_get_input_s(input_s, tempGreen_s_pitch, tlsx-1, tlsy) + 
			k0_get_input_s(input_s, tempGreen_s_pitch, tlsx+1, tlsy)) / 2.0f + 
			(2.0f * k0_get_input_s(input_s, tempGreen_s_pitch, tlsx, tlsy) - 
			k0_get_input_s(input_s, tempGreen_s_pitch, tlsx-2, tlsy) - 
			k0_get_input_s(input_s, tempGreen_s_pitch, tlsx+2, tlsy)) / 4.0f;
	
		// calculate diagonal green
		valD =  (k0_get_input_s(input_s, tempGreen_s_pitch, tlsx-1, tlsy) + 
			k0_get_input_s(input_s, tempGreen_s_pitch, tlsx+1, tlsy) + 
			k0_get_input_s(input_s, tempGreen_s_pitch, tlsx, tlsy-1) + 
			k0_get_input_s(input_s, tempGreen_s_pitch, tlsx, tlsy+1)) / 4.0f + 
			(4.0f * k0_get_input_s(input_s, tempGreen_s_pitch, tlsx, tlsy) - 
			k0_get_input_s(input_s, tempGreen_s_pitch, tlsx-2, tlsy) - 
			k0_get_input_s(input_s, tempGreen_s_pitch, tlsx+2, tlsy) -
			k0_get_input_s(input_s, tempGreen_s_pitch, tlsx, tlsy-2) -
			k0_get_input_s(input_s, tempGreen_s_pitch, tlsx, tlsy+2)) / 8.0f;

	}

	__syncthreads();

	// once computation result is saved in register, we must put them back in shared memory, to ensure coalesced store
	// we will try to achieve 64-bit coalesced store for CC 1.0 and 1.1, so the trade-off will be 2-way bank conflict

	// address to put and then read the computation result into shared memory
	//tlsx = (threadIdx.x) % 8 + (threadIdx.y / 8) * 8;
	//tlsy = threadIdx.x / 8 + 2 * (threadIdx.y % 8);

	tlsx = (threadIdx.x) % (blockDim.x / 2) + (threadIdx.y / (blockDim.y / 2)) * (blockDim.x / 2);
	tlsy = threadIdx.x / (blockDim.x / 2) + 2 * (threadIdx.y % (blockDim.y / 2));

	// address to write into global memory
	tltx = blockDim.x * blockIdx.x + threadIdx.x;
	tlty = blockDim.y * blockIdx.y + threadIdx.y;

	if (tltx < (params.width + params.window_radius) / 2 && tlty < (params.height + params.window_radius))
	{

		// put valV into shared memory
		k0_set_input_s(input_s, staging_pitch, tlsx, tlsy, valV);

		__syncthreads();
	
		set_tempGreen(tempGreenV_g, tempGreen_g_pitch, tltx, tlty, k0_get_input_s(input_s, staging_pitch, threadIdx.x, threadIdx.y));


		// put valH itu shared memory
		k0_set_input_s(input_s, staging_pitch, tlsx, tlsy, valH);

		__syncthreads();

		set_tempGreen(tempGreenH_g, tempGreen_g_pitch, tltx, tlty, k0_get_input_s(input_s, staging_pitch, threadIdx.x, threadIdx.y));
		

		// put valD into shared memory
		k0_set_input_s(input_s, staging_pitch, tlsx, tlsy, valD);

		__syncthreads();

		set_tempGreen(tempGreenD_g, tempGreen_g_pitch, tltx, tlty, k0_get_input_s(input_s, staging_pitch, threadIdx.x, threadIdx.y));
		
	}
}

void calculate_temp_green(float* input_g, size_t input_g_pitch, 
	float* tempGreenV_g, float* tempGreenH_g, float* tempGreenD_g,
	size_t tempGreen_g_pitch, vcd_params params, int filter, cudaDeviceProp prop, vcd_sf_mp_kernel_times* times)
{
	dim3 dimBlock, dimGrid;
	size_t smemPitch, stagingPitch, smemSize;
	int mod = 0;
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
		dimBlock.x = 16;
		banks = 16;
	}
	dimBlock.y = 16;

	dimGrid.x = (int) ceil((float)((params.width + params.window_radius) / 2) / (float)dimBlock.x);
	dimGrid.y = (int) ceil((float)((params.height + params.window_radius) / (float)dimBlock.y));
	smemPitch = ((2 * dimBlock.x + 2 * params.temp_green_radius) * sizeof(float)); // assuming the size of element is float
	stagingPitch = (dimBlock.x + banks / 2) * sizeof(float);
	smemSize = smemPitch * (dimBlock.y + 2 * params.temp_green_radius);

	cudaEventRecord(start,0);
	calculate_temp_green_k<<<dimGrid,dimBlock,smemSize>>>(input_g, 
		input_g_pitch, tempGreenV_g, tempGreenH_g, tempGreenD_g, 
		tempGreen_g_pitch, smemPitch, stagingPitch, params, filter);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&kerntime, start, stop);
	times->calc_temp_green = kerntime;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

}

__global__ void calculate_e_k(int* e, size_t eptc, float* input, size_t inputptc, size_t smemptc, size_t stgptc, vcd_params params, int filter)
{
	// load from input to shared memory. However, the position is not aligned for coalescing.
	// so, we must calculate the offset to achieve coalescing
	int sx, sy, tx, ty, cx, cy, color, val = 0;
	float ev = 0.0f;
	int maxrad = max(params.window_radius + params.temp_green_radius, params.e_radius);
	int offsety = maxrad > params.e_radius? maxrad - params.e_radius : 0;
	
	extern __shared__ float smem[];

	// source address
	sx = 2 * blockDim.x * blockIdx.x;
	sy = blockDim.y * blockIdx.y + offsety;

	int loopx = (int)ceil((float)(2 * blockDim.x + params.e_radius + maxrad) / (float)(blockDim.x));
	int loopy = (int)ceil((float)(blockDim.y + 2 * params.e_radius) / (float)(blockDim.y));

	for (int y = 0; y < loopy; y++)
	{
		for (int x = 0; x < loopx; x++)
		{
			int nsx = sx + x * blockDim.x + threadIdx.x;
			int nsy = sy + y * blockDim.y + threadIdx.y;

			int endx = sx + 2 * blockDim.x + params.e_radius + maxrad;
			int endy = sy + blockDim.y + 2 * params.e_radius;

			if (nsx < endx && nsy < endy && nsx < (params.width + 2 * maxrad) && nsy < (params.height + 2 * maxrad))
			{
				tx = x * blockDim.x + threadIdx.x;
				ty = y * blockDim.y + threadIdx.y;

				smem[ty * (smemptc / sizeof(float)) + tx] = input[nsy * (inputptc / sizeof(float)) + nsx];
			}
		}
	}

	//sx = 2 * (threadIdx.x % 8) + (threadIdx.y / 8) * 16 + maxrad; // x address for loading from input
	//sy = threadIdx.x / 8 + 2 * (threadIdx.y % 8) + params.e_radius; // y address for loading from input

	sx = 2 * (threadIdx.x % (blockDim.x / 2)) + (threadIdx.y / (blockDim.y / 2)) * blockDim.x + maxrad; // x address for loading from input
	sy = threadIdx.x / (blockDim.x / 2) + 2 * (threadIdx.y % (blockDim.y / 2)) + params.e_radius; // y address for loading from input
	
	cx = params.x + 2 * blockDim.x * blockIdx.x + sx;
	cy = params.y + blockDim.y * blockIdx.y + sy;

	color = fc(filter, cx, cy);
	if (color == COLOR_GREEN_RED)
	{
		sx += params.grtor;
		cx += params.grtor;
	} else if (color == COLOR_GREEN_BLUE)
	{
		sx += params.gbtob;
		cx += params.gbtob;
	}

	float Lh, Lv = 0.0f;
	
	if (cx < (params.x + params.width) && cy < (params.y + params.height))
	{
		for (int dx = -2; dx < 3; dx++)
		{
			for (int dy = -2; dy < 3; dy++)
			{
				/*if (dx != 0 || dy != 0)
				{
					Lh += abs(k0_get_input_s(smem, smemptc, sx+dx,sy+dy)-k0_get_input_s(smem, smemptc, sx,sy+dy));
					Lv += abs(k0_get_input_s(smem, smemptc, sx+dx,sy+dy)-k0_get_input_s(smem, smemptc, sx+dx,sy));
				}*/

				if (dx != 0)
				{
					Lv += abs(k0_get_input_s(smem, smemptc, sx+dx,sy+dy)-k0_get_input_s(smem, smemptc, sx+dx,sy));
				}

				if (dy != 0)
				{
					Lh += abs(k0_get_input_s(smem, smemptc, sx+dx,sy+dy)-k0_get_input_s(smem, smemptc, sx,sy+dy));
				}

			}

		}

		ev = max((float)Lv/(float)Lh,(float)Lh/(float)Lv);
		val = 0;

		if (ev > params.e_threshold)
		{
			// sharp block, insert previous g computation result
			if (Lh < Lv)
			{
				val = HORZ_EDGE_PRE_CALC;
			} else
			{
				val = VERT_EDGE_PRE_CALC;
			}
		} else
		{
			val = TEX_PRE_CALC;
		}
	}

	__syncthreads(); // make sure all threads done the calculation, because after this the shared memory content will be wiped out

	// put the result into shared memory. 2-way bank conflict
	//sx = (threadIdx.x) % 8 + (threadIdx.y / 8) * 8;
	//sy = 2 * (threadIdx.y % 8) + (threadIdx.x / 8);

	sx = (threadIdx.x) % (blockDim.x / 2) + (threadIdx.y / (blockDim.y / 2)) * (blockDim.x / 2);
	sy = threadIdx.x / (blockDim.x / 2) + 2 * (threadIdx.y % (blockDim.y / 2));

	((int*)smem)[sy * (stgptc / sizeof(int)) + sx] = val;

	__syncthreads();

	// put the result in shared memory into global memory
	tx = blockDim.x * blockIdx.x + threadIdx.x;
	ty = blockDim.y * blockIdx.y + threadIdx.y;

	if (tx < (params.width / 2) && ty < params.height)
	{
		e[ty * (eptc / sizeof(int)) + tx] = ((int*)smem)[threadIdx.y * (stgptc / sizeof(int)) + threadIdx.x];
	}
	
}

__global__ void test_e_k(float* outr, size_t outrptc, int* e, size_t eptc, vcd_params params, int filter)
{
	int sx, sy, tx, ty, cx, cy, color = 0;
	
	// source address
	sx = blockDim.x * blockIdx.x + threadIdx.x;
	sy = blockDim.y * blockIdx.y + threadIdx.y;

	// target address
	tx = 2 * sx;
	ty = sy;

	// color address
	cx = params.x + tx;
	cy = params.y + ty;

	// adjust source and target position if it falls into red/blue pixel
	color = fc(filter, cx, cy);
	if (color == COLOR_GREEN_RED)
	{
		tx += correct_x(params, COLOR_RED, color);
		//tx += params.grtor;
	} else if (color == COLOR_GREEN_BLUE)
	{
		tx += correct_x(params, COLOR_BLUE, color);
		//tx += params.gbtob;
	}

	if (sx < (params.width / 2) && sy < params.height)
	{
		// copy from input to output
		int val = e[sy * (eptc / sizeof(int)) + sx];
		if (val == HORZ_EDGE_PRE_CALC || val == VERT_EDGE_PRE_CALC)
			outr[ty * (outrptc / sizeof(unsigned int)) + tx] = 8191.0f;
	}
}

void calculate_e(int* e, size_t eptc, float* input, size_t inputptc, vcd_params params, int filter, cudaDeviceProp prop, vcd_sf_mp_kernel_times* times)
{
	int maxrad = max(params.window_radius + params.temp_green_radius, params.e_radius);

	float kerntime = 0.0f;
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 dimBlock;
	dim3 dimGrid;
	int banks = 0;

	if (prop.major == 2)
	{
		dimBlock.x = 32;
		banks = 32;
	}
	else
	{
		dimBlock.x = 16;
		banks = 16;
	}

	dimBlock.y = 16;

	dimGrid.x = (int)ceil((float)(params.width / 2) / (float)dimBlock.x);
	dimGrid.y = (int)ceil((float)params.height / (float)dimBlock.y);

	size_t smemptc, stgptc = 0;
	size_t smemsz= 0;
	int mod = 0;
	smemptc = get_smem_pitch(((params.e_radius + maxrad + 2 * dimBlock.x) * sizeof(float)));
	stgptc = (dimBlock.x + banks / 2) * sizeof(float);
	smemsz = smemptc * (2 * params.e_radius + dimBlock.y);

	cudaEventRecord(start, 0);
	calculate_e_k<<<dimGrid, dimBlock, smemsz>>>(e, eptc, input, inputptc, smemptc, stgptc, params, filter);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&kerntime, start, stop);

	times->calc_e = kerntime;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

__global__ void test_temp_green_k(float* outg, size_t outg_ptc, const float* tempg, size_t tempg_ptc, 
	const float* input, size_t input_ptc, vcd_params params, int filter)
{

	// first, copy from input to output
	int sx, sy, tx, ty, cx, cy, color = 0;
	int maxrad = max(params.window_radius + params.temp_green_radius, params.e_radius);

	// source address
	sx = 2 * (blockDim.x * blockIdx.x + threadIdx.x) + maxrad;
	sy = blockDim.y * blockIdx.y + threadIdx.y + maxrad;

	// target address
	tx = sx - maxrad;
	ty = sy - maxrad;

	// color address
	cx = params.x + tx;
	cy = params.y + ty;

	// adjust source and target position if it falls into red/blue pixel
	color = fc(filter, cx, cy);
	if (color == COLOR_RED)
	{
		sx -= params.grtor;
		tx -= params.grtor;
	} else if (color == COLOR_BLUE)
	{
		sx += params.gbtob;
		tx += params.gbtob;
	}

	if (tx < params.width && ty < params.height)
	{
		// copy from input to output
		outg[ty * (outg_ptc / sizeof(unsigned int)) + tx] = (unsigned int)input[sy * (input_ptc / sizeof(float)) + sx];
	}

	// second, copy from temp green to output

	// source address. Dont forget to offset it by the size of window radius
	sx = blockDim.x * blockIdx.x + threadIdx.x;
	sy = blockDim.y * blockIdx.y + threadIdx.y;

	// target address
	tx = 2 * (blockDim.x * blockIdx.x + threadIdx.x);
	ty = blockDim.y * blockIdx.y + threadIdx.y;

	cx = params.x + tx;
	cy = params.y + ty;

	color = fc(filter, cx, cy);

	if (color == COLOR_GREEN_RED)
	{
		tx += params.grtor;
	} else if (color == COLOR_GREEN_BLUE)
	{
		tx += params.gbtob;
	}

	if (tx < params.width && ty < params.height)
	{
		// copy from temp green to output
		outg[ty * (outg_ptc / sizeof(unsigned int)) + tx] = (unsigned int)tempg[sy * (tempg_ptc / sizeof(float)) + sx];
	}
}


void test_temp_green(float* outg, size_t outg_ptc, const float* tempg, size_t tempg_ptc, 
	const float* input, size_t input_ptc, vcd_params params, int filter, cudaDeviceProp prop)
{
	dim3 dimBlock;
	dim3 dimGrid;

	if (prop.major == 2)
		dimBlock.x = 32;
	else // prop.major = 1.x
		dimBlock.x = 16;
	dimBlock.y = 16;

	dimGrid.x = (int)ceil((float)params.width / (float)dimBlock.x);
	dimGrid.y = (int)ceil((float)params.height / (float)dimBlock.y);

	test_temp_green_k<<<dimGrid, dimBlock>>>(outg, outg_ptc, tempg, tempg_ptc, input, input_ptc, params, filter);
}

void test_e(float* outr, size_t outrptc, int* e, size_t eptc, vcd_params params, int filter, cudaDeviceProp prop)
{
	dim3 dimBlock;
	dim3 dimGrid;
	
	if (prop.major == 2)
		dimBlock.x = 32;
	else // prop.major = 1.x
		dimBlock.x = 16;
	dimBlock.y = 16;

	dimGrid.x = (int)ceil((float)(params.width / 2) / (float)dimBlock.x);
	dimGrid.y = (int)ceil((float)params.height / (float)dimBlock.y);

	test_e_k<<<dimGrid, dimBlock>>>(outr, outrptc, e, eptc, params, filter);
}

