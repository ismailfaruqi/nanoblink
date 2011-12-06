#include "gpu_stdafx.h"
#include "gpu_common.cuh"
#include <cutil.h>

// kernel forward declarations

// this kernel just copies G and (R/B) values, and deduce the rest by averaging neighbor pixel
template<class OutputType>
__global__ void gpu_test_d(OutputType* dest, 
						   size_t dest_pitch_bytes, 
						   size_t dest_width_bytes,
						   size_t dest_height,
						   float* src,
						   size_t src_pitch_bytes,
						   size_t src_width_bytes,
						   size_t src_height,
						   int filter)
{
	// copy one pixel to all pixel element, so it results grayscale image
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int x = blockDim.x * blockIdx.x + tx; // global x address
	int y = blockDim.y * blockIdx.y + ty; // global y address

	int w = dest_width_bytes / (sizeof(OutputType)*4);
	int h = dest_height;

	// make sure it is within bound	
	if (x < w && y < h)
	{
		dest[y * dest_pitch_bytes/sizeof(OutputType) + 4 * x + 0] = src[y*src_pitch_bytes/sizeof(float)+x]; // BLUE.
		dest[y * dest_pitch_bytes/sizeof(OutputType) + 4 * x + 1] = src[y*src_pitch_bytes/sizeof(float)+x]; // GREEN
		dest[y * dest_pitch_bytes/sizeof(OutputType) + 4 * x + 2] = src[y*src_pitch_bytes/sizeof(float)+x]; // REED
		dest[y * dest_pitch_bytes/sizeof(OutputType) + 4 * x + 3] = 8191;
		
	}
}

template <class OutputType>
void gpu_test(int device, // the device will be used to perform demosaicing
			  OutputType* output, // output memory address in host
			  size_t stride,
			  int outputWidth, // output memory width
			  int outputHeight, // output memory height
			  unsigned short *librawInput, // input memory address in host
			  int inputWidth, // input width
			  int inputHeight, // input height
			  int filter, // filter
			  void* pinnedMem,//, // page-locked memory address
			  LibRaw* libRaw
			  //int option // option
			  )
{
	// calculate global memory requirement for RAW image in the device
	size_t inpsz = inputWidth * inputHeight * sizeof(float);

	// calculate global memory requirement for result image in the device
	size_t outsz = outputWidth * outputHeight * 4 * sizeof(OutputType);

	// copy RAW image from unsigned short to float
	float* rawimg = (float*) malloc (inpsz); // new operator doesn't seems to work?
	for (int y = 0; y < inputHeight; y++)
	{
		for (int x = 0; x < inputWidth; x++)
		{
			rawimg[y*inputWidth+x] = (float)librawInput[y*inputWidth+x];
		}		
	}

	// allocate RAW image in global device memory, using pitch
	// it's better to allocate different memory for input than accessing output directly, to coalesce memory access
	float* rawimg_d = NULL;
	size_t rawimg_d_ptc = 0;
	size_t rawimg_d_w = inputWidth * sizeof(float);

	// allocate raw image on device
	CUDA_SAFE_CALL(cudaMallocPitch((void**)&rawimg_d, &rawimg_d_ptc, rawimg_d_w, inputHeight));
	CUDA_SAFE_CALL(cudaMemset2D(rawimg_d, rawimg_d_ptc, 0, rawimg_d_w, inputHeight));
	
	// allocate result image in global device memory, using pitch
	// result image is in RGBA, 64 bit.
	unsigned short* result_d = NULL;
	size_t result_d_ptc = 0;
	size_t result_d_width = inputWidth * 4 * sizeof(OutputType);

	CUDA_SAFE_CALL(cudaMallocPitch((void**)&result_d, &result_d_ptc, result_d_width, outputHeight));
	CUDA_SAFE_CALL(cudaMemset2D(result_d, result_d_ptc, 0, result_d_width, outputHeight));
	
	// copy RAW image to global device memory, using cudaMemcpy2D
	CUDA_SAFE_CALL(cudaMemcpy2D(rawimg_d, rawimg_d_ptc, rawimg, rawimg_d_w, rawimg_d_w, inputHeight, cudaMemcpyHostToDevice));
	
	//// launch kernel
	dim3 dimBlock(16, 16);
	dim3 dimGrid(outputWidth / dimBlock.x, outputHeight / dimBlock.y);
	gpu_test_d<unsigned short> <<<dimGrid, dimBlock>>> (result_d, result_d_ptc, result_d_width, outputHeight,rawimg_d, rawimg_d_ptc, rawimg_d_w, inputHeight, filter);
	
	// copy back result from device to host
	CUDA_SAFE_CALL(cudaMemcpy2D(output, stride, result_d, result_d_ptc, result_d_width, outputHeight, cudaMemcpyDeviceToHost));
	
	// free device global memory
	CUDA_SAFE_CALL(cudaFree(rawimg_d));
	CUDA_SAFE_CALL(cudaFree(result_d));
	
	// free host memory
	free(rawimg);

}

template void gpu_test<unsigned char>(int device, // the device will be used to perform demosaicing
			  unsigned char* output, // output memory address in host
			  size_t stride,
			  int outputWidth, // output memory width
			  int outputHeight, // output memory height
			  unsigned short *librawInput, // input memory address in host
			  int inputWidth, // input width
			  int inputHeight, // input height
			  int filter, // filter
			  void* pinnedMem,
			  LibRaw* libRaw//, // page-locked memory address
			  //int option // option
			  );

template void gpu_test<unsigned short>(int device, // the device will be used to perform demosaicing
			  unsigned short* output, // output memory address in host
			  size_t stride,
			  int outputWidth, // output memory width
			  int outputHeight, // output memory height
			  unsigned short *librawInput, // input memory address in host
			  int inputWidth, // input width
			  int inputHeight, // input height
			  int filter, // filter
			  void* pinnedMem,
			  LibRaw* libRaw//, // page-locked memory address
			  //int option // option
			  );