#ifndef GPU_VCD_SF_H
#define GPU_VCD_SF_H

// entry point for c/c++ applicatons.
// output: B-G-R pixel output, each element is 16-bit, must be allocated beforehand.
// librawInput: 4-pixel array input, must be allocated beforehand. Memory layout must be contiguous.
// rawWidth: width of raw image
// rawHeight: height of raw image
void gpu_vcd_sf_singlegpu_singlepass(int device, unsigned short* output, int outputWidth, int outputHeight,
				unsigned short (*librawInput)[4], int inputWidth, int inputHeight, int filters,
				void* pinnedMem);

#endif