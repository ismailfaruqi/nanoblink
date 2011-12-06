#ifndef VCD_COMMON_H
#define VCD_COMMON_H

typedef struct {
	float e_threshold;
	unsigned int e_radius;
	unsigned int temp_green_radius;
	unsigned int window_radius;
	unsigned int x;
	unsigned int y;
	int grtor;
	int gbtob;
	int grtob;
	int gbtor;
	size_t width;
	size_t height;
	size_t dimx;
	size_t dimy;
} vcd_params;

typedef struct {
	float host_to_dev;
	float convert_input;
	float calc_e;
	float calc_temp_green;
	float separate_input;
	float calc_temp_var_pos;
	float calc_var_neg;
	float calc_rest;
	float convert_output;
	float dev_to_host;
} vcd_sf_mp_kernel_times;

void vcd_params_init(vcd_params* params,
	float thres,
	unsigned int erad,
	unsigned int tempgrad,
	unsigned int winrad,
	unsigned int x,
	unsigned int y,
	size_t width,
	size_t height,
	size_t dimx,
	size_t dimy,
	int xycolor);

int vcd_get_maxrad(vcd_params params);

#endif