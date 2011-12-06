#include <cuda.h>
#include <cuda_runtime.h>

#ifndef GPU_MEMGETINFO
#define GPU_MEMGETINFO

void gpu_memgetinfo(int dev, unsigned int* freeMem, unsigned int *totalMem)
{
	float *f = NULL;
	cudaMalloc((void**)&f, sizeof(float));
	unsigned int fm = 0;
	unsigned int mm = 0;
	cuMemGetInfo(&fm, &mm);
	*freeMem = fm;
	*totalMem = mm;
	cudaFree(f);	
}

#endif