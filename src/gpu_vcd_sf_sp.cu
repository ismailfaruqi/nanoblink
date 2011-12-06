#include "gpu_vcd_sf_sp.h"
#include "gpu_common.cuh"
#include <windows.h>

bool print = false;

// shared memory
struct gpu_vcd_sf_kernel_onepass_param
{
	size_t vgi_s_size;
	size_t hgi_s_size;
	size_t bgi_s_size;
	size_t output_s_size;
	int loadsPerThread;
	int filters;
	int isDownSweep;
};

// compute green element
__global__ void gpu_vcd_sf_onepass_kernel(const int loopIdx, 
										  const unsigned short (*librawInput_g)[4],
										  unsigned short* output_g,
										  const int stride,
										  const int height,
										  const gpu_vcd_sf_kernel_onepass_param params);

// this is the official version
__device__ inline unsigned short output_s_get(unsigned short(*output_s)[3], int x, int y, int filter, gpu_vcd_sf_kernel_onepass_param param);

// this is the official version
__device__ inline void output_s_set(unsigned short(*output_s)[3], int x, int y, unsigned short color, unsigned short val, gpu_vcd_sf_kernel_onepass_param param);

__device__ inline unsigned short gi_s_get(unsigned short* gi_s, int x, int y, gpu_vcd_sf_kernel_onepass_param param);

__device__ inline void gi_s_set(unsigned short* gi_s, int x, int y, unsigned short val, gpu_vcd_sf_kernel_onepass_param param);

__device__ void interpolate_g_on_rb(unsigned short (*output_s)[3], unsigned short* hgi_s, unsigned short* vgi_s, unsigned short* bgi_s, 
									int blockX, int blockY, gpu_vcd_sf_kernel_onepass_param params, int loopIdx, bool isDownSweep);

//compares if the float f1 is equal with f2 and returns 1 if true and 0 if false
int compare_float(float f1, float f2, float precision)
{
	if (((f1 - precision) < f2) && 
		((f1 + precision) > f2))
	{
		return 1;
	}
	else
	{
		return 0;
	}
}


void gpu_vcd_sf_singlegpu_singlepass(int device, unsigned short* output, int outputWidth, int outputHeight,
				unsigned short (*librawInput)[4], int rawWidth, int rawHeight, int filter,
				void* pinnedMem)
{
	// set device
	cudaSetDevice(device);

	int processedRows = 0;

	// raw dimension and output dimension must be same on each axis
	// RESERVED FOR HALF-SIZE VERSION
	/*float widthRatio = (float)outputWidth/(float)rawWidth;
	float heightRatio = (float)outputHeight/(float)rawHeight;
	if (!compare_float(widthRatio,heightRatio, 0.0001f))
	{
		printf("ERROR: Width and height ratio are not same! Width ratio = %.02f, Height ratio = %.02f\n", widthRatio, heightRatio);
		return;
	}*/

	// block dimension
	dim3 dimBlock(8,8);

	// set shared memory size
	gpu_vcd_sf_kernel_onepass_param params;
	params.loadsPerThread = 2;
	params.bgi_s_size = (sizeof(unsigned short) * params.loadsPerThread * ((dimBlock.x + 4) / 2) * (dimBlock.y + 4));
	params.vgi_s_size = (sizeof(unsigned short) * params.loadsPerThread * ((dimBlock.x + 4) / 2) * (dimBlock.y + 4));
	params.hgi_s_size = (sizeof(unsigned short) * params.loadsPerThread * ((dimBlock.x + 4) / 2) * (dimBlock.y + 4));
	params.output_s_size = (sizeof(unsigned short) * 3 * (params.loadsPerThread * dimBlock.x + 10) * (dimBlock.y + 10)); // sizeof output
	params.filters = filter;

	size_t smemSize = 0;
	smemSize += params.vgi_s_size; // sizeof vgi_s
	smemSize += params.bgi_s_size;
	smemSize += params.hgi_s_size;
	smemSize += params.output_s_size;
	
	while (processedRows < (outputHeight - 4))
	{
		// cudaMalloc cuMemGetInfo hack.
		float *f = NULL;
		CUDA_SAFE_CALL(cudaMalloc((void**)&f, sizeof(float)));
		CUDA_SAFE_CALL(cudaFree(f));

		// get device memory size
		unsigned int freeDeviceGlobalMemory = 0;
		unsigned int maxDeviceGlobalMemory = 0;
		cudaDeviceProp prop;

		// cuMemGetInfo will return 0 in emulation mode, so get memory info from device properties instead.
#ifdef __DEVICE_EMULATION__
		
		// there is only device number 0 in EmuDebug mode
		cudaGetDeviceProperties(&prop, 0);
		freeDeviceGlobalMemory = prop.totalGlobalMem;
		maxDeviceGlobalMemory = prop.totalGlobalMem;
#else
		cudaGetDeviceProperties(&prop, device);
		cuMemGetInfo(&freeDeviceGlobalMemory, &maxDeviceGlobalMemory);
#endif		

		int totalMemSizePerHeight = sizeof(unsigned short) * (4 * rawWidth + 3 * outputWidth);

		// get iterations
		int rowToProcess = min((rawHeight - processedRows),freeDeviceGlobalMemory/totalMemSizePerHeight);
		
		size_t inputMemSize = 4 * rawWidth * rowToProcess * sizeof(unsigned short);
		size_t outputMemSize = 3 * outputWidth * rowToProcess * sizeof(unsigned short);

		printf("Available memory: %d; used memory: %d\n", freeDeviceGlobalMemory, inputMemSize+outputMemSize);

		unsigned short (*librawInput_g)[4] = NULL;
		unsigned short* output_g = NULL;
		
		CUDA_SAFE_CALL(cudaMalloc((void**)&librawInput_g, inputMemSize));
		CUDA_SAFE_CALL(cudaMalloc((void**)&output_g, outputMemSize));
		
		// copy input from CPU to GPU
		CUDA_SAFE_CALL(cudaMemcpy(librawInput_g, (librawInput + (processedRows * rawWidth * 4  * sizeof(unsigned short))), inputMemSize, cudaMemcpyHostToDevice));

		int maxBlockX = (rawWidth - 10) / (params.loadsPerThread * dimBlock.x);
		int maxBlockY = (rowToProcess - 10) / (dimBlock.y);

		int i = 0;
		while (i < maxBlockY)
		{
			int dimGridX = min(i+1,maxBlockX);
			dim3 dimGrid(dimGridX);
			params.isDownSweep = true; // down sweep

			gpu_vcd_sf_onepass_kernel<<<dimGrid, dimBlock, smemSize>>>(i, librawInput_g, output_g, rawWidth, rowToProcess, params);
			
			cudaThreadSynchronize();
			
			cudaError_t err = cudaGetLastError();
			if (err != cudaSuccess)
				printf("Down-sweep: Loop[%d], dimGridX = %d, Error: %s\n", i, dimGridX, cudaGetErrorString(err));

			i++;
		}
		
		// right-sweep
		i = 0;
		while (i < (maxBlockX - 1))
		{
			int dimGridX = min(maxBlockY,maxBlockX-i-1);
			dim3 dimGrid(dimGridX);
			params.isDownSweep = false; // right-sweep

			gpu_vcd_sf_onepass_kernel<<<dimGrid, dimBlock, smemSize>>>(i, librawInput_g, output_g, rawWidth, rowToProcess, params);

			cudaThreadSynchronize();

			cudaError_t err = cudaGetLastError();
			if (err != cudaSuccess)
				printf("Right-sweep: Loop[%d], dimGridX = %d, Error: %s\n", i, dimGridX, cudaGetErrorString(err));

			i++;
		}
		
		// copy output from GPU to CPU
		CUDA_SAFE_CALL(cudaMemcpy((output + (processedRows * rawWidth * sizeof(unsigned short) * 3)), output_g, outputMemSize, cudaMemcpyDeviceToHost));

		// free memory on GPU
		CUDA_SAFE_CALL(cudaFree(librawInput_g));
		CUDA_SAFE_CALL(cudaFree(output_g));

		processedRows += (rowToProcess - 4);
	}
	
}

// this kernel computes all
// loads per thread: 1 or 2
__global__ void gpu_vcd_sf_onepass_kernel(const int loopIdx, 
										  const unsigned short (*librawInput_g)[4],
										  unsigned short* output_g,
										  const int stride,
										  const int height,
										  const gpu_vcd_sf_kernel_onepass_param params)
{

	extern __shared__ unsigned short smem[];

	unsigned short (*output_s)[3] = (unsigned short(*)[3])&smem[0];
	unsigned short* vgi_s = (unsigned short*)(&(output_s[params.output_s_size/(sizeof(unsigned short)*3)][0]));
	unsigned short* hgi_s = (unsigned short*)(&vgi_s[params.vgi_s_size/sizeof(unsigned short)]); // a chunk of temporary horizontal green interpolation in shared memory
	unsigned short* bgi_s = (unsigned short*)(&hgi_s[params.hgi_s_size/sizeof(unsigned short)]); // a chunk of temporary diagonal green interpolation in shared memory
	
	int blockX = params.isDownSweep? blockIdx.x : loopIdx + blockIdx.x + 1;
	int blockY = params.isDownSweep? loopIdx - blockIdx.x : ((height - 10) / blockDim.y) - blockIdx.x - 1;

	int linearTid = threadIdx.y * blockDim.x + threadIdx.x;
	int output_sStride = params.loadsPerThread * blockDim.x + 10;

	// load librawInput_g and output_g to shared memory. This algorithm is not coalesced
	for (
		int i = linearTid; 
		i < output_sStride*(blockDim.y + 10); 
		i += (blockDim.x * blockDim.y))
	{
		int newTidX = i % output_sStride;
		int newTidY = i / output_sStride;
		
		int globalX = blockX * (params.loadsPerThread * blockDim.x) + newTidX;
		int globalY = blockY * (blockDim.y) + newTidY;
		int inputLinear = globalY * stride + globalX;

		int color = fc(params.filters, globalX, globalY);

		for (int k = 0; k < 3; k++)
		{
			// load from librawInput_g
			if (k == color || (k == 1 && color == COLOR_GREEN_BLUE))
			{
				output_s[i][2-k] = librawInput_g[inputLinear][color];
			} else // load from output_g
			{
				output_s[i][2-k] = output_g[3*inputLinear+2-k];
			}
		}
		
	}

	__syncthreads();

	int green = 0;
	
	linearTid = 2 * (threadIdx.y * blockDim.x + threadIdx.x);
	// A. compute temporary green interpolation at R/B position
	for (
		int i = linearTid;
		i < (params.loadsPerThread * blockDim.x + 4)*(blockDim.y + 4);
		i += (2 * blockDim.x * blockDim.y)
		)
	{
		int x = i % (params.loadsPerThread * blockDim.x + 4);
		int y = i / (params.loadsPerThread * blockDim.x + 4);

		int globalX = params.loadsPerThread * (blockDim.x * blockX) + x + 4;
		int globalY = blockY * blockDim.y + y + 4;
		int colorCntr = fc(params.filters, globalX, globalY);
		
		if (!(colorCntr == COLOR_RED || colorCntr == COLOR_BLUE))
		{
			globalX++; x++;
			colorCntr = fc(params.filters, globalX, globalY);
		}
		int colorVert = COLOR_GREEN;
		int colorHorz = COLOR_GREEN;
		
		int green = 0;

		green = 
			((
			output_s_get(output_s, x-1, y, colorHorz, params) + 
			output_s_get(output_s, x+1, y, colorHorz, params)
			)  / 2) + 
			((
			2 * output_s_get(output_s, x, y, colorCntr, params) - 
			output_s_get(output_s, x-2, y, colorCntr, params) - 
			output_s_get(output_s, x+2, y, colorCntr, params)
			) / 4);

		gi_s_set(hgi_s,x,y,green,params);

		// A.2 compute vertical temp green
		green = 
			((
			output_s_get(output_s, x, y-1, colorVert, params) + 
			output_s_get(output_s, x, y+1, colorVert, params)
			) / 2) + 
			((
			2 * output_s_get(output_s, x, y, colorCntr, params) - 
			output_s_get(output_s, x, y-2, colorCntr, params) - 
			output_s_get(output_s, x, y+2, colorCntr, params)
			) / 4);

		gi_s_set(vgi_s,x,y,green,params);

		// A.3 compute diagonal temp green
		green = 
			((
			output_s_get(output_s, x-1, y, colorHorz, params) +  
			output_s_get(output_s, x+1, y, colorHorz, params) +
			output_s_get(output_s, x, y-1, colorVert, params) + 
			output_s_get(output_s, x, y+1, colorVert, params)) / 4) + 
			((
			4 * output_s_get(output_s, x, y, colorCntr, params) - 
			output_s_get(output_s, x-2, y, colorCntr, params) - 
			output_s_get(output_s, x+2, y, colorCntr, params) - 
			output_s_get(output_s, x, y-2, colorCntr, params) - 
			output_s_get(output_s, x, y+2, colorCntr, params)) / 8);

		gi_s_set(bgi_s,x,y,green,params);

	}

	__syncthreads();
	
	// B. compute G component on B/R position diagonally

	// do it down-sweep
	for (int i = 0; i < blockDim.y; i++)
	{
		if ((threadIdx.x + blockDim.x * threadIdx.y) < min(i+1,blockDim.x))
		{
			interpolate_g_on_rb(output_s, hgi_s, vgi_s, bgi_s, blockX, blockY, params, i, true);
		}
		__syncthreads();
	}

	__syncthreads();

	// do it right-sweep
	for (int i = 0; i < (blockDim.x - 1); i++)
	{
		if ((threadIdx.x + blockDim.x * threadIdx.y) < min(blockDim.y,blockDim.x-i-1))
		{
			interpolate_g_on_rb(output_s, hgi_s, vgi_s, bgi_s, blockX, blockY, params, i, false);
		}
		__syncthreads();
	}	

	// C. compute B and R component on G position
	int x = params.loadsPerThread * threadIdx.x;
	int y = threadIdx.y;
	int globalX = params.loadsPerThread * blockDim.x * blockX + x + 4;
	int globalY = (blockDim.y * blockY + y + 4);
	int colorCntr = fc(params.filters, globalX, globalY);
	if (!(colorCntr == COLOR_GREEN_BLUE || colorCntr == COLOR_GREEN_RED))
	{
		x++; globalX++;
		colorCntr = fc(params.filters, globalX, globalY);
	}
	int colorVert = (colorCntr + 1) % 4;
	int colorHorz = colorCntr - 1; 
	// if pixType == GR, c0 = R and c1 = B.
	// if pixType == GB, c0 = B and c1 = G.
	int c[2];
	c[0] = output_s_get(output_s, x,y,COLOR_GREEN,params) + ((
		output_s_get(output_s, x-1,y,colorHorz,params) -
		output_s_get(output_s, x-1,y,COLOR_GREEN,params) +
		output_s_get(output_s, x+1,y,colorHorz, params) - 
		output_s_get(output_s, x+1,y,COLOR_GREEN,params)
	) / 2);

	c[1] = output_s_get(output_s, x,y,COLOR_GREEN,params) + ((
		output_s_get(output_s, x,y+1,colorVert,params) -
		output_s_get(output_s, x,y+1,COLOR_GREEN,params) + 
		output_s_get(output_s, x,y-1,colorVert,params) - 
		output_s_get(output_s, x,y-1,COLOR_GREEN,params) 
	)  / 2);

	if (colorCntr == COLOR_GREEN_RED)
	{
		output_s_set(output_s, x, y, COLOR_RED, c[0], params); // colorCtr = 1 -> c[0], 3->1
		output_s_set(output_s, x, y, COLOR_BLUE, c[1], params); // 1->1, 3->0

	} else if (colorCntr == COLOR_GREEN_BLUE)
	{
		output_s_set(output_s, x, y, COLOR_RED, c[1], params); // colorCtr = 1 -> c[0], 3->1
		output_s_set(output_s, x, y, COLOR_BLUE, c[0], params); // 1->1, 3->
	}
	
	__syncthreads();

	// D. compute B/R component on R/B position
	x = params.loadsPerThread * threadIdx.x;
	globalX = params.loadsPerThread * blockDim.x * blockX + x + 4;
	globalY = (blockDim.y * blockY + y + 4);
	colorCntr = fc(params.filters, globalX, globalY);
	if (!(colorCntr == COLOR_RED || colorCntr == COLOR_BLUE))
	{
		x++; globalX++;
		colorCntr = fc(params.filters, globalX, globalY);
	}
	int colorDiag = 2 - colorCntr;
	int sigma = 0;
//#ifdef __DEVICE_EMULATION__
//	printf("R/B->B/R:Central(%d,%d),R=%d,G=%d,B=%d\n",globalX,globalY,
//		output_s_get(output_s,x,y,COLOR_RED,params),
//		output_s_get(output_s,x,y,COLOR_GREEN,params),
//		output_s_get(output_s,x,y,COLOR_BLUE,params));
//#endif
	
	for (int m = -1; m < 2; m += 2)
	{
		for (int w = -1; w < 2; w += 2)
		{
			
//#ifdef __DEVICE_EMULATION__
//			print = true;
//			printf("R/B->B/R:(%d,%d):(%d,%d)ABS(%d,%d)LOC(%d,%d)->(%d,%d),colorDiag=[%d]%d,colorGreen=%d\n\n",globalX,globalY,w,m,globalX+w,globalY+m,
//				x,y,x+w,y+m,colorDiag,
//			output_s_get(output_s,x+w,y+m,colorDiag,params),
//			output_s_get(output_s,x+w,y+m,COLOR_GREEN,params));
//			print = false;
//#endif
			sigma += (output_s_get(output_s,x+w,y+m,colorDiag,params) - output_s_get(output_s, x+w, y+m, COLOR_GREEN, params));
		}
	}
	sigma /= 4;
	sigma += output_s_get(output_s, x, y, COLOR_GREEN, params);

//#ifdef __DEVICE_EMULATION__
//	if ((threadIdx.x > (blockDim.x - 4)) && colorDiag == COLOR_RED)
//		printf("RED(%d,%d):%d\n",globalX,globalY,sigma);
//#endif
	output_s_set(output_s, x, y, colorDiag, sigma, params); // 0->2, 2->0
		
	__syncthreads();
	
	x = params.loadsPerThread * threadIdx.x;
	for (int i = 0; i < params.loadsPerThread; i++)
	{
		int target = (blockDim.y * blockY + y + 4) * stride + params.loadsPerThread * blockDim.x * blockX + x + 4 + i;

		output_g[3*target] = output_s_get(output_s, x+i, y, COLOR_BLUE, params);
		output_g[3*target+1] = output_s_get(output_s, x+i, y, COLOR_GREEN, params);
		output_g[3*target+2] = output_s_get(output_s, x+i, y, COLOR_RED, params);

	}

}

// TESTED: OK
__device__ inline unsigned short output_s_get(unsigned short(*output_s)[3], int x, int y, int filter, gpu_vcd_sf_kernel_onepass_param param)
{
//#ifdef __DEVICE_EMULATION__
//	if (print)
//	{
//		printf("output_s get:(%d,%d)[%d]\n", x, y, filter);
//	}
//#endif
	
	return output_s[(y + 4) * (param.loadsPerThread * blockDim.x + 10) + x + 4][2-filter];
}

__device__ inline unsigned short gi_s_get(unsigned short* gi_s, int x, int y, gpu_vcd_sf_kernel_onepass_param param)
{
	return gi_s[y * (param.loadsPerThread * (blockDim.x + 4) / 2) + (x/2)];
}

__device__ inline void gi_s_set(unsigned short* gi_s, int x, int y, unsigned short val, gpu_vcd_sf_kernel_onepass_param param)
{
	gi_s[y * (param.loadsPerThread * (blockDim.x + 4) / 2) + (x/2)] = val;
}

__device__ inline void output_s_set(unsigned short(*output_s)[3], int x, int y, unsigned short color, unsigned short val, gpu_vcd_sf_kernel_onepass_param param)
{
	output_s[(y + 4) * (param.loadsPerThread * blockDim.x + 10) + x + 4][2-color] = val;
}

__device__ void interpolate_g_on_rb(unsigned short (*output_s)[3], unsigned short* hgi_s, unsigned short* vgi_s, unsigned short* bgi_s, 
									int blockX, int blockY, gpu_vcd_sf_kernel_onepass_param params, int loopIdx, bool isDownSweep)
{
	int x = isDownSweep ? (params.loadsPerThread * threadIdx.x) : ((loopIdx + threadIdx.x + 1) * params.loadsPerThread);
	int y = isDownSweep ? (loopIdx - threadIdx.x) : (blockDim.y - 1 - threadIdx.x);

	int globalX = params.loadsPerThread * (blockDim.x * blockX + x) + 4;
	int globalY = blockY * blockDim.y + y + 4;
	int colorCntr = fc(params.filters, globalX, globalY);

	if (!(colorCntr == COLOR_RED || colorCntr == COLOR_BLUE))
	{
		globalX++; x++;
		colorCntr = fc(params.filters, globalX, globalY);
	}
	int colorDiag = 2 - colorCntr;
	int colorVert = COLOR_GREEN;
	int colorHorz = COLOR_GREEN;

	int gi_sLinIdx = y * (params.loadsPerThread * (blockDim.x + 4) / 2) + x / 2;
	int green = 0;

	// calculate LH and LV first
	int Lv = 0; int Lh = 0;

	for (int dx = -2; dx < 3; dx++)
	{
		for (int dy = -2; dy < 3; dy++)
		{
			if (dx != 0 || dy != 0)
			{
				if ((dx&1)&&(dy&1)) // dx is odd, dy is odd
				{
					Lh += abs(output_s_get(output_s, x+dx,y+dy,colorDiag, params)-output_s_get(output_s, x,y+dy,colorVert, params));
					Lv += abs(output_s_get(output_s, x+dx,y+dy,colorDiag, params)-output_s_get(output_s, x+dx,y,colorHorz, params));
				} else if (!(dx&1)&&(dy&1)) // dx is even, dy is odd
				{
					Lh += abs(output_s_get(output_s, x+dx,y+dy,colorVert, params)-output_s_get(output_s, x,y+dy,colorVert, params));
					Lv += abs(output_s_get(output_s, x+dx,y+dy,colorVert, params)-output_s_get(output_s, x+dx,y,colorCntr, params));
				} else if ((dx&1)&&!(dy&1)) // dx is odd, dy is even
				{
					Lh += abs(output_s_get(output_s, x+dx,y+dy,colorHorz, params)-output_s_get(output_s, x,y+dy,colorCntr, params));
					Lv += abs(output_s_get(output_s, x+dx,y+dy,colorHorz, params)-output_s_get(output_s, x+dx,y,colorHorz, params));
				} else // dx is even, dy is even
				{
					Lh += abs(output_s_get(output_s, x+dx,y+dy,colorCntr, params)-output_s_get(output_s, x,y+dy,colorCntr, params));
					Lv += abs(output_s_get(output_s, x+dx,y+dy,colorCntr, params)-output_s_get(output_s, x+dx,y,colorCntr, params));
				}
			}

		}

	}

	float e = max((float)Lv/(float)Lh,(float)Lh/(float)Lv);

	if (e > 2.0f)
	{
		// sharp block, insert previous g computation result
		if (Lh < Lv)
		{
			green = hgi_s[gi_sLinIdx];
			//output_s[3*target+1] = hgi_s[gi_sLinIdx];
		} else
		{
			green = vgi_s[gi_sLinIdx];
			//output_s[3*target+1] = vgi_s[gi_sLinIdx];
		}
	} else
	{
		// pattern block
		int tempv[10];
		int temph[10];
		int tempbv[10];
		int tempbh[10];

		for (int i = -4; i < 5; i+=2)
		{
			if (i < 0)
			{
				tempv[i+4] = output_s_get(output_s,x,y+i,colorCntr,params) + output_s_get(output_s, x, y+i, COLOR_GREEN, params);//[3*((y+i)*(outputWidth)+x)+1];
				temph[i+4] = output_s_get(output_s,x+i,y,colorCntr,params) + output_s_get(output_s, x+i, y, COLOR_GREEN, params);//[3*((y)*(outputWidth)+x+i)+1];
				tempbv[i+4] = tempv[i+4];
				tempbh[i+4] = temph[i+4];
			} else if (i >= 0)
			{
				tempv[i+4] = output_s_get(output_s,x,y+i,colorCntr,params) + gi_s_get(vgi_s, x, y+i, params); //vgi_s[(y+i)*giWidth+(x>>1)];
				temph[i+4] = output_s_get(output_s,x+i,y,colorCntr,params) + gi_s_get(hgi_s, x+i, y, params); //hgi_s[(y)*giWidth+((x+i)>>1)];
				tempbv[i+4] = output_s_get(output_s,x,y+i,colorCntr,params) + gi_s_get(bgi_s, x, y+i, params); //bgi_s[(y+i)*giWidth+(x>>1)];
				tempbh[i+4] = output_s_get(output_s,x+i,y,colorCntr,params) + gi_s_get(bgi_s, x+i, y, params); //bgi_s[(y)*giWidth+((x+i)>>1)];
			}								
		}

		for (int i = -3; i < 4; i+=2)
		{
			tempv[i+4] = (tempv[i+3]+tempv[i+5])>>1;
			temph[i+4] = (temph[i+3]+temph[i+5])>>1;
			tempbv[i+4] = (tempbv[i+3]+tempbv[i+5])>>1;
			tempbh[i+4] = (tempbh[i+3]+tempbh[i+5])>>1;
		}

		tempv[9] = 0; temph[9] = 0; tempbv[9] = 0; tempbh[9] = 0;

		// calculate average on final element
		for (int i = 0; i < 0; i++)
		{
			tempv[9] += tempv[i];
			temph[9] += temph[i];
			tempbv[9] += tempbv[i];
			tempbh[9] += tempbh[i];
		}

		tempv[9] /= 9;
		temph[9] /= 9;
		tempbv[9] /= 9;
		tempbh[9] /= 9;

		// calculate sigma
		double sigmav = 0.0; double sigmah = 0.0; 
		double sigmabv = 0.0; double sigmabh = 0.0;
		double sigmab = 0.0;

		for (int i = 0; i < 9; i++)
		{
			sigmav += pow((tempv[i]-tempv[9]), 2.0);
			sigmah += pow((temph[i]-temph[9]), 2.0);
			sigmabv += pow((tempbv[i]-tempbv[9]), 2.0);
			sigmabh += pow((tempbh[i]-tempbh[9]), 2.0);
		}

		sigmav /= 9.0;
		sigmah /= 9.0;
		sigmab = 0.5 * (sigmabv / 9.0 + sigmabh / 9.0);

		if ((sigmav < sigmah) && (sigmav < sigmab))
		{
			green = gi_s_get(vgi_s, x, y, params);
			//pixels[3*(y*outputWidth+x)+1] = vgi[(y)*giWidth+(x>>1)];
		} else if ((sigmah < sigmav) && (sigmah < sigmab))
		{
			green = gi_s_get(hgi_s, x, y, params);
			//pixels[3*(y*outputWidth+x)+1] = hgi[(y)*giWidth+(x>>1)];
		} else if ((sigmab < sigmav) && (sigmab < sigmah))
		{
			green = gi_s_get(bgi_s, x, y, params);
			//pixels[3*(y*outputWidth+x)+1] = bgi[(y)*giWidth+(x>>1)];
		}
	}

	output_s_set(output_s, x, y, COLOR_GREEN, green, params);
//#ifdef __DEVICE_EMULATION__
//	printf("R/B->G:P(%d,%d)R=%d,G=%d,B=%d\n",globalX,globalY,
//		output_s_get(output_s, x, y, COLOR_RED, params),
//		output_s_get(output_s, x, y, COLOR_GREEN, params),
//		output_s_get(output_s, x, y, COLOR_BLUE, params));
//#endif
}