/* -*-c++-*- */

#ifndef CUDA_REFINED_CURV_RATIO_HH
#define CUDA_REFINED_CURV_RATIO_HH

#include "curv_types.h"

void find_refined_curv_ratio_cycle(const float* image, int xDim, int yDim, int zDim,
 				   float length_weight, float curv_power,
				   CurvEstimatorType curv_type, bool grad_abs, float* out_image);

#endif