#include "vcd_common.h"
#include "common.h"

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
	int xycolor)
{
	params->e_threshold = thres;
	params->e_radius = erad;
	params->temp_green_radius = tempgrad;
	params->window_radius = winrad;
	params->x = x;
	params->y = y;
	params->width = width;
	params->height = height;
	params->dimx = dimx;
	params->dimy = dimy;
	params->grtor = (xycolor == COLOR_RED || xycolor == COLOR_GREEN_BLUE? -1 : 1);
	params->gbtob = (xycolor == COLOR_RED || xycolor == COLOR_GREEN_BLUE? 1 : -1);
	params->grtob = (xycolor == COLOR_RED || xycolor == COLOR_GREEN_RED? 1 : -1);
	params->gbtor = (xycolor == COLOR_RED || xycolor == COLOR_GREEN_RED? -1 : 1);
	
}

int vcd_get_maxrad(vcd_params params)
{
	return (params.temp_green_radius + params.window_radius > params.e_radius? params.temp_green_radius + params.window_radius : params.e_radius);
}