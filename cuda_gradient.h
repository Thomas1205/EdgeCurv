#ifndef CUDA_GRADIENT_H
#define CUDA_GRADIENT_H

float2 gradient(const float* image, int x, int y, int xDim, int yDim);

float2 gradient(const float* image, int x, int y, int z, int xDim, int yDim, int zDim);

float2 subpixel_gradient(const float* image, float x, float y, int xDim, int yDim);

#endif
