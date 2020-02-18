/* -*-c++-*- */

#ifndef CUDA_RATIO_MATH_CUH
#define CUDA_RATIO_MATH_CUH

inline __device__ bool mul_is_negative(const int in1, const int in2);
	   
inline __device__ uint2 umul32(const uint in1, const uint in2);

inline __device__ bool dist_lower(const int old_num, const int old_denom, const int new_num, const int new_denom, 
           	   	 	  const int ratio_num, const int ratio_denom, int* status); 	     

#include "cuda_ratio_math.cu"

#endif