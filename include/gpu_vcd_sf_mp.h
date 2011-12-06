/****************************************************
* Accelerated VCD Algorithm
* File Name: gpu_vcd_sf_mp.h
* Contents:
* Constants for VCD algorithm implementation on GPU
* Author: Muhammad Ismail Faruqi
*****************************************************/
#pragma once

#include "vcd_common.h"

// algorithm option
#define COALESCED		0x1
#define NONCOALESCED	0x0
#define UNOPTIMIZED		0x0

template <class T>
void gpu_vcd_sf_singlegpu_multipass(int device, unsigned short* output, int stride, int outputWidth, int outputHeight,
									   unsigned short *librawInput, int inputWidth, int inputHeight, int filter, vcd_params params,
									   vcd_sf_mp_kernel_times* times,
									   void* pinnedMem);

template <class T>
void gpu_vcd_sf_singlegpu_multipass(int device, // the device will be used to perform demosaicing
									unsigned short* output, // output memory address in host
									int stride,
									int outputWidth, // output memory width
									int outputHeight, // output memory height
									unsigned short* librawInput, // input memory address in host
									int inputWidth, // input width
									int inputHeight, // input height
									int filter, // filter
									vcd_params params,
									vcd_sf_mp_kernel_times* times,
									void* pinnedMem, // page-locked memory address
									int option // option
									);