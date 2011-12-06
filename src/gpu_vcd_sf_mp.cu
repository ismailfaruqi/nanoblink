// file: gpu_vcd_sf_mp.cu
// multi-pass version of VCD kernel
#include "gpu_common.cuh"
#include "gpu_vcd_common.cuh"
#include "gpu_io.cuh"
#include "gpu_vcd_sf_mp.h"
#include "gpu_vcd_sf_mp_k0.cuh"
#include "gpu_vcd_sf_mp_k1.cuh"
#include "gpu_vcd_sf_mp_k1_v2.cuh"
#include "gpu_vcd_sf_mp_k2.cuh"
#include <math.h>

size_t get_global_memory_available(int device);

//////////////////////////////////////////////////////////////////////////
/// Kernel parameters
//////////////////////////////////////////////////////////////////////////

// parameter for kernel #3
struct gpu_vcd_sf_multipass_calc_red_blue_param
{
};

////////////////////////////////////////////////////////////////////////// END OF KERNEL PARAMETERS



//////////////////////////////////////////////////////////////////////////
/// Kernel forward declarations
//////////////////////////////////////////////////////////////////////////


// this kernel calculates:
// 1. blue AND red channel in green position
// 2. blue OR red channel in red or blue position
__global__ void gpu_vcd_sf_multipass_calc_red_blue_kernel(
	unsigned short (*librawInput_g)[4],
	const gpu_vcd_sf_multipass_calc_red_blue_param param
	);

////////////////////////////////////////////////////////////////////////// END OF KERNEL FORWARD DECLARATIONS

template <typename T>
void gpu_vcd_sf_singlegpu_multipass(int device, unsigned short* output, int stride, int outputWidth, int outputHeight,
									unsigned short *librawInput, int inputWidth, int inputHeight, int filter, vcd_params params,
									vcd_sf_mp_kernel_times* times,
									void* pinnedMem)
{
	gpu_vcd_sf_singlegpu_multipass<T>(device, output, stride, outputWidth, outputHeight,
									librawInput, inputWidth, inputHeight, filter, params, times, pinnedMem, UNOPTIMIZED);
}

__global__ void test_fill_red(unsigned int* out, size_t outptc, vcd_params params)
{
	int tx, ty = 0;

	tx = blockIdx.x * blockDim.x + threadIdx.x;
	ty = blockIdx.y * blockDim.y + threadIdx.y;

	if (tx < params.width && ty < params.height)
		out[ty * outptc / sizeof(unsigned int) + tx] = 8191;
}

template <typename T>
void gpu_vcd_sf_singlegpu_multipass(int device, unsigned short* output, int stride, int outputWidth, int outputHeight,
									unsigned short *librawInput, int inputWidth, int inputHeight, int filter, vcd_params params,
									vcd_sf_mp_kernel_times* times, void* pinnedMem, int option)
{
	// init kernel time
	times->convert_input = 0.0f;
	times->convert_output = 0.0f;
	times->calc_e = 0.0f;
	times->calc_temp_green = 0.0f;
	times->calc_temp_var_pos = 0.0f;
	times->calc_var_neg = 0.0f;

	// set device
	cudaSetDevice(device);
	cudaDeviceProp prop;

	// get device properties
	cudaGetDeviceProperties(&prop, device);

	cudaEvent_t start;
	cudaEvent_t stop;
	float kerntime = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	// block dimension
	dim3 dimBlock;
	// grid dimension
	dim3 dimGrid;

	size_t smemSize = 0;
	int mod = 0;
	size_t smemPitch = 0;

	// a variable to keep track processed row
	int processedRows = 0;

	// available global memory
	size_t gmemSize = get_global_memory_available(device);
		
	/// CALCULATE GLOBAL MEMORY SIZE REQUIREMENTS FOR 1 ROW
	int totalApronRadius = max((params.window_radius + params.temp_green_radius),params.e_radius);

	// memory requirement for ushort-typed input
	unsigned short *input4_g = NULL;
	size_t input4Pitch = 0;
	size_t input4MemSizePerRow = (params.width + 2 * totalApronRadius) * sizeof(unsigned short);
	CUDA_SAFE_CALL(cudaMallocPitch((void**)&input4_g, &input4Pitch, input4MemSizePerRow, 1));
	CUDA_SAFE_CALL(cudaFree(input4_g));

	// memory requirement for float-typed input
	float* librawInput_g = NULL; // input raw data
	size_t librawInputPitch = 0;
	size_t inputMemSizePerRow = (params.width + 2 * totalApronRadius) * sizeof(float);
	CUDA_SAFE_CALL(cudaMallocPitch((void**)&librawInput_g, &librawInputPitch, inputMemSizePerRow, 1));
	CUDA_SAFE_CALL(cudaFree(librawInput_g));

	// memory requirement for per-channel 1 tuple output/input
	float* test;
	size_t inchPitch = 0;
	size_t inchMemSizePerRow = (params.width + totalApronRadius) / 2 * sizeof(float);
	CUDA_SAFE_CALL(cudaMallocPitch((void**)&test, &inchPitch, inchMemSizePerRow, 1));
	CUDA_SAFE_CALL(cudaFree(test));

	// memory requirement for per-channel output
	float* outputR_g = NULL;
	float* outputG_g = NULL;
	float* outputB_g = NULL;
	size_t outputPitch = 0;
	size_t outputMemSizePerRow = (params.width) * sizeof(float);
	CUDA_SAFE_CALL(cudaMallocPitch((void**)&outputR_g, &outputPitch, outputMemSizePerRow, 1));
	CUDA_SAFE_CALL(cudaFree(outputR_g));

	// memory requirement for output
	unsigned short* output_g = NULL;
	size_t output_g_pitch = 0;
	size_t outputGMemSizePerRow = (params.width) * 4 * sizeof(unsigned short);
	CUDA_SAFE_CALL(cudaMallocPitch((void**)&output_g, &output_g_pitch, outputGMemSizePerRow, 1));
	CUDA_SAFE_CALL(cudaFree(output_g));

	// memory requirement for temporary green channel
	float *tempGreenH_g, *tempGreenV_g, *tempGreenD_g = NULL; // temporary green horizontal, vertical, and diagonal
	size_t tempGreenPitch = 0; // temporary green pitch size, it will be same for horizontal, vertical and diagonal
	size_t tempGreenMemSizePerRow = (int)ceil((float)(params.width + params.window_radius) / 2.0f) * sizeof(float); // it will only calculate area that falls within e calculation area
	CUDA_SAFE_CALL(cudaMallocPitch((void**)&tempGreenH_g, &tempGreenPitch, tempGreenMemSizePerRow, 1));
	CUDA_SAFE_CALL(cudaFree(tempGreenH_g));

	// memory requirement for e values
	int* eValues_g = NULL;
	size_t eValuesPitch = 0;
	size_t eValuesMemSizePerRow = (int)ceil((float)params.width / 2.0f) * sizeof(int);
	CUDA_SAFE_CALL(cudaMallocPitch((void**)&eValues_g, &eValuesPitch, eValuesMemSizePerRow, 1));
	CUDA_SAFE_CALL(cudaFree(eValues_g));

	// memory requirement for temporary variance values
	float* tempVarH_g, *tempVarV_g, *tempVarDH_g, *tempVarDV_g = NULL;
	size_t tempVarPitch = 0;
	size_t tempVarMemSizePerRow = (int)ceil((float)params.width / 2.0f) * sizeof(float);
	CUDA_SAFE_CALL(cudaMallocPitch((void**)&tempVarH_g, &tempVarPitch, tempVarMemSizePerRow, 1));
	CUDA_SAFE_CALL(cudaFree(tempVarH_g));

	// now calculate memory size per row
	size_t calcmem = 0;
	calcmem += (3 * outputPitch + output_g_pitch + 3 * tempGreenPitch + eValuesPitch + 4 * tempVarPitch + 12 * inchPitch); // TODO: change after testing separate_input_k
	size_t memsizePerRow = (input4Pitch > librawInputPitch? input4Pitch : librawInputPitch) + calcmem;

	// substract for apron memory requirement
	gmemSize -= (2 * totalApronRadius * input4Pitch);
	gmemSize -= (2 * totalApronRadius * librawInputPitch);
	gmemSize -= (params.window_radius * 3 * tempGreenPitch);
	gmemSize -= (12 * totalApronRadius * inchPitch);
	
	// number of row calculable per loop
	int rows = gmemSize / memsizePerRow;

	// make number of row calculable even
	if (rows % 2 == 1)
		rows--;

	while (processedRows < params.height)
	{
		vcd_params newparams = params;
		newparams.y = params.y + processedRows;
		newparams.height = min(params.height - processedRows, rows);

		// determine memory transfer sizes
		transfer_params trparams;
		trparams.left_radius = min(totalApronRadius, newparams.x);
		trparams.top_radius = min(totalApronRadius, newparams.y);
		trparams.right_radius = min(totalApronRadius, newparams.dimx - (newparams.x + newparams.width));
		trparams.bottom_radius = min(totalApronRadius, newparams.dimy - (newparams.y + newparams.height));
		
		// copy 4 tuple input from CPU to GPU
		int input4Height = newparams.height + 2 * totalApronRadius;
		CUDA_SAFE_CALL(cudaMalloc((void**)&input4_g, input4Pitch * input4Height));

		cudaEventRecord(start, 0);

		cudaError_t err = cudaMemcpy2D(
			((char*)input4_g) + (totalApronRadius - trparams.top_radius) * input4Pitch + (totalApronRadius - trparams.left_radius) * sizeof(unsigned short), // target transfer address
			input4Pitch, // pitch
			&librawInput[(newparams.y - trparams.top_radius) * newparams.dimx + (newparams.x - trparams.left_radius)],
			newparams.dimx * sizeof(unsigned short), 
			(newparams.width + trparams.left_radius + trparams.right_radius) * sizeof(unsigned short), 
			(newparams.height + trparams.top_radius + trparams.bottom_radius),
			cudaMemcpyHostToDevice);

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&kerntime, start, stop);
		times->host_to_dev = kerntime;
		
		if (err != cudaSuccess)
			printf("Error: %s\n", cudaGetErrorString(err));

		// copy converted input array from host to device
		CUDA_SAFE_CALL(cudaMalloc((void**)&librawInput_g, librawInputPitch * (newparams.height + 2 * totalApronRadius)));
		CUDA_SAFE_CALL(cudaMemset2D(librawInput_g, librawInputPitch, 0, inputMemSizePerRow, (newparams.height + 2 * totalApronRadius))); // pad with zero

		// launch input converter kernel
		convert_input(librawInput_g, librawInputPitch, input4_g, input4Pitch, filter, newparams, prop, times);
		
		// free temporary input4 in global memory
		CUDA_SAFE_CALL(cudaFree(input4_g));

		// allocate output memory in device
		CUDA_SAFE_CALL(cudaMalloc((void**)&outputR_g, outputPitch * newparams.height));
		CUDA_SAFE_CALL(cudaMalloc((void**)&outputG_g, outputPitch * newparams.height));
		CUDA_SAFE_CALL(cudaMalloc((void**)&outputB_g, outputPitch * newparams.height));
		CUDA_SAFE_CALL(cudaMemset2D(outputR_g, outputPitch, 0, outputMemSizePerRow, newparams.height));
		CUDA_SAFE_CALL(cudaMemset2D(outputG_g, outputPitch, 0, outputMemSizePerRow, newparams.height));
		CUDA_SAFE_CALL(cudaMemset2D(outputB_g, outputPitch, 0, outputMemSizePerRow, newparams.height));

		// TEMPORARY GREEN CHANNEL
		// allocate memory for storing temporary green
		size_t tempGreenHeight = newparams.height + params.window_radius;
		CUDA_SAFE_CALL(cudaMalloc((void**)&tempGreenH_g, tempGreenPitch * tempGreenHeight));
		CUDA_SAFE_CALL(cudaMalloc((void**)&tempGreenV_g, tempGreenPitch * tempGreenHeight));
		CUDA_SAFE_CALL(cudaMalloc((void**)&tempGreenD_g, tempGreenPitch * tempGreenHeight));
	
		// fill temporary green array with zero
		CUDA_SAFE_CALL(cudaMemset2D(tempGreenH_g, tempGreenPitch, 0, tempGreenMemSizePerRow, tempGreenHeight));
		CUDA_SAFE_CALL(cudaMemset2D(tempGreenV_g, tempGreenPitch, 0, tempGreenMemSizePerRow, tempGreenHeight));
		CUDA_SAFE_CALL(cudaMemset2D(tempGreenD_g, tempGreenPitch, 0, tempGreenMemSizePerRow, tempGreenHeight));

		// launch the kernel to calculate temporary green
		calculate_temp_green(librawInput_g, 
			librawInputPitch, tempGreenV_g, tempGreenH_g, tempGreenD_g, 
			tempGreenPitch, newparams, filter, prop, times);

		// next, allocate memory for storing e value for calculated red / blue pixel only
		CUDA_SAFE_CALL(cudaMalloc((void**)&eValues_g, eValuesPitch * newparams.height));
		
		// launch the kernel to calculate e values
		calculate_e(eValues_g, eValuesPitch, librawInput_g, librawInputPitch, newparams, filter, prop, times);

		// allocate RR, BB, GRG, GRB
		size_t inchHeight = (newparams.height + 2 * totalApronRadius) / 2;
		float* inch_g[12];
		for (int i = 0; i < 12; i++)
		{
			inch_g[i] = NULL;
		}

		CUDA_SAFE_CALL(cudaMalloc((void**)&inch_g[RR], inchPitch * inchHeight));
		CUDA_SAFE_CALL(cudaMalloc((void**)&inch_g[BB], inchPitch * inchHeight));
		CUDA_SAFE_CALL(cudaMalloc((void**)&inch_g[GRG], inchPitch * inchHeight));
		CUDA_SAFE_CALL(cudaMalloc((void**)&inch_g[GBG], inchPitch * inchHeight));
		// memset
		CUDA_SAFE_CALL(cudaMemset2D(inch_g[RR], inchPitch, 0, inchPitch, inchHeight));
		CUDA_SAFE_CALL(cudaMemset2D(inch_g[BB], inchPitch, 0, inchPitch, inchHeight));
		CUDA_SAFE_CALL(cudaMemset2D(inch_g[GRG], inchPitch, 0, inchPitch, inchHeight));
		CUDA_SAFE_CALL(cudaMemset2D(inch_g[GBG], inchPitch, 0, inchPitch, inchHeight));

		// launch the kernel to separate input from combined unsigned short into per-channel float
		separate_input(inch_g[RR], inch_g[BB], inch_g[GRG], inch_g[GBG], inchPitch, librawInput_g, librawInputPitch, filter, newparams, prop, times);

		// input
		CUDA_SAFE_CALL(cudaFree(librawInput_g));

		// launch the kernel to calculate green values at red and blue filter position
		/*CUDA_SAFE_CALL(cudaMalloc((void**)&tempVarH_g, tempVarPitch * newparams.height));
		CUDA_SAFE_CALL(cudaMalloc((void**)&tempVarV_g, tempVarPitch * newparams.height));
		CUDA_SAFE_CALL(cudaMalloc((void**)&tempVarDH_g, tempVarPitch * newparams.height));
		CUDA_SAFE_CALL(cudaMalloc((void**)&tempVarDV_g, tempVarPitch * newparams.height));*/

		// allocate BG and RG
		CUDA_SAFE_CALL(cudaMalloc((void**)&inch_g[RG], inchPitch * inchHeight));
		CUDA_SAFE_CALL(cudaMalloc((void**)&inch_g[BG], inchPitch * inchHeight));
		// memset
		CUDA_SAFE_CALL(cudaMemset2D(inch_g[RG], inchPitch, 0, inchPitch, inchHeight));
		CUDA_SAFE_CALL(cudaMemset2D(inch_g[BG], inchPitch, 0, inchPitch, inchHeight));

		// launch the kernel to calculate variance element in positive direction

		// old version
	/*	interpolate_g_dg_pos(outputR_g, outputPitch, tempVarH_g, tempVarV_g, tempVarDH_g, tempVarDV_g, tempVarPitch, 
			tempGreenH_g, tempGreenV_g, tempGreenD_g, tempGreenPitch,
			librawInput_g, librawInputPitch,
			eValues_g, eValuesPitch,
			filter, newparams, prop, times);

		stream_g_calculation(outputR_g, outputG_g, outputB_g, outputPitch, 
			tempVarH_g, tempVarV_g, tempVarDH_g, tempVarDV_g, tempVarPitch,
			tempGreenH_g, tempGreenV_g, tempGreenD_g, tempGreenPitch,
			librawInput_g, librawInputPitch, 
			eValues_g, eValuesPitch,
			filter, newparams, prop, times);*/	

		// new version
	/*	interpolate_g_dg_pos(outputR_g, outputPitch, tempVarH_g, tempVarV_g, tempVarDH_g, tempVarDV_g, tempVarPitch, 
			tempGreenH_g, tempGreenV_g, tempGreenD_g, tempGreenPitch,
			inch_g[RR], inch_g[BB], inchPitch,
			eValues_g, eValuesPitch,
			filter, newparams, prop, times);*/
		
		stream_g_calculation(inch_g[RG], inch_g[BG], inchPitch, 
			tempVarH_g, tempVarV_g, tempVarDH_g, tempVarDV_g, tempVarPitch,
			tempGreenH_g, tempGreenV_g, tempGreenD_g, tempGreenPitch,
			inch_g[RR], inch_g[BB],  inchPitch, 
			eValues_g, eValuesPitch,
			filter, newparams, prop, times);

		//
		//test_temp_green(outputG_g, outputPitch, tempGreenH_g, tempGreenPitch, librawInput_g, librawInputPitch, newparams, filter, prop);

		//test_e(outputR_g, outputPitch, eValues_g, eValuesPitch, newparams, filter, prop);

		// free mems

		// temporary variance
		/*CUDA_SAFE_CALL(cudaFree(tempVarH_g));
		CUDA_SAFE_CALL(cudaFree(tempVarV_g));
		CUDA_SAFE_CALL(cudaFree(tempVarDH_g));
		CUDA_SAFE_CALL(cudaFree(tempVarDV_g));*/

		// temporary greens
		CUDA_SAFE_CALL(cudaFree(tempGreenH_g));
		CUDA_SAFE_CALL(cudaFree(tempGreenV_g));
		CUDA_SAFE_CALL(cudaFree(tempGreenD_g));

		// e values
		CUDA_SAFE_CALL(cudaFree(eValues_g));

		// allocate the rest of channels
		CUDA_SAFE_CALL(cudaMalloc((void**)&inch_g[GRR], inchPitch * inchHeight));
		CUDA_SAFE_CALL(cudaMalloc((void**)&inch_g[GRB], inchPitch * inchHeight));
		CUDA_SAFE_CALL(cudaMalloc((void**)&inch_g[GBR], inchPitch * inchHeight));
		CUDA_SAFE_CALL(cudaMalloc((void**)&inch_g[GBB], inchPitch * inchHeight));
		CUDA_SAFE_CALL(cudaMalloc((void**)&inch_g[BR], inchPitch * inchHeight));
		CUDA_SAFE_CALL(cudaMalloc((void**)&inch_g[RB], inchPitch * inchHeight));
		
		calculate_rest(inch_g[GRR], inch_g[GRB], inch_g[GBR], inch_g[GBB], inch_g[BR], inch_g[RB],
					   inch_g[GRG], inch_g[GBG], inch_g[RG], inch_g[BG], inch_g[RR], inch_g[BB], inchPitch, filter, newparams, prop, times);
		
		// create output memory
		CUDA_SAFE_CALL(cudaMalloc((void**)&output_g, output_g_pitch * newparams.height));
		
		// launch output converter kernel

		cudaEventRecord(start, 0);

		//convert_output(output_g, output_g_pitch, outputR_g, outputG_g, outputB_g, outputPitch, newparams, prop, times);
		combine_channels(output_g, output_g_pitch, inch_g[RR], inch_g[RG], inch_g[RB], 
			inch_g[BR], inch_g[BG], inch_g[BB], 
			inch_g[GRR], inch_g[GRG], inch_g[GRB], 
			inch_g[GBR], inch_g[GBG], inch_g[GBB], inchPitch, filter, newparams, prop, times);

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&kerntime, start, stop);
		times->dev_to_host = kerntime;

		// free output inch
		for (int i = 0; i < 12; i++)
		{
			if (inch_g[i] != NULL)
				CUDA_SAFE_CALL(cudaFree(inch_g[i]));
		}
		
		// copy output in global memory to local memory
		CUDA_SAFE_CALL(cudaMemcpy2D((char*)output + newparams.y * stride + newparams.x * 4 * sizeof(unsigned short), stride,
			output_g, output_g_pitch,
			outputGMemSizePerRow,
			newparams.height,
			cudaMemcpyDeviceToHost));

		CUDA_SAFE_CALL(cudaFree(output_g));
	
		// free output
		CUDA_SAFE_CALL(cudaFree(outputR_g));
		CUDA_SAFE_CALL(cudaFree(outputG_g));
		CUDA_SAFE_CALL(cudaFree(outputB_g));
		
		processedRows += newparams.height;
	}

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaThreadSynchronize();
	
}

// C++ template hack: float
template void gpu_vcd_sf_singlegpu_multipass<float>(int device, unsigned short* output, int stride, int outputWidth, int outputHeight,
									unsigned short *librawInput, int inputWidth, int inputHeight, int filter, vcd_params params, vcd_sf_mp_kernel_times* times,
									void* pinnedMem, int option);

template void gpu_vcd_sf_singlegpu_multipass<float>(int device, unsigned short* output, int stride, int outputWidth, int outputHeight,
									   unsigned short *librawInput, int inputWidth, int inputHeight, int filter, vcd_params params, vcd_sf_mp_kernel_times* times,
									   void* pinnedMem);

// C++ template hack: unsigned short
template void gpu_vcd_sf_singlegpu_multipass<unsigned short>(int device, unsigned short* output, int stride, int outputWidth, int outputHeight,
									unsigned short *librawInput, int inputWidth, int inputHeight, int filter, vcd_params params, vcd_sf_mp_kernel_times* times,
									void* pinnedMem, int option);

template void gpu_vcd_sf_singlegpu_multipass<unsigned short>(int device, unsigned short* output, int stride, int outputWidth, int outputHeight,
									   unsigned short *librawInput, int inputWidth, int inputHeight, int filter, vcd_params params, vcd_sf_mp_kernel_times* times,
									   void* pinnedMem);



size_t get_global_memory_available(int device)
{
	float* f = NULL;
	cudaMalloc((void**)&f, sizeof(float));
	cudaFree(f);
	
	unsigned int freeDeviceGlobalMemory = 0;
	unsigned int maxDeviceGlobalMemory = 0;
	cuMemGetInfo(&freeDeviceGlobalMemory, &maxDeviceGlobalMemory);

	return freeDeviceGlobalMemory;
}