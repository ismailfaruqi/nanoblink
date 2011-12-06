#include "gpu_common.cuh"

size_t get_smem_pitch(size_t sz)
{
	int device, int divisor;
	cudaDeviceProp prop;
	
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&prop, device);

	if (prop.major == 2)
		divisor = 128;
	else
		divisor = 64;

	size_t result = sz;
	int mod = 0;
	mod = sz % divisor;
	if (mod > 0)
		result += (divisor - mod);
	return result;
}

