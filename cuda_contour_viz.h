#ifndef CUDA_CONTOUR_VIZ_H
#define CUDA_CONTOUR_VIZ_H

void mark_contour(int* segmentation, int xDim, int yDim, const int2* explicit_contour, uint contour_length,
		  int width = 2, bool clear = true);

#endif
