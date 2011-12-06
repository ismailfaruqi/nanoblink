#pragma once

#include <libraw.h>

// a test for cudaMemcpy2D. Basically just perform copy. Calculations are performed in float
template <class OutputType>
void gpu_test(	int device, // the device will be used to perform demosaicing
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
				);