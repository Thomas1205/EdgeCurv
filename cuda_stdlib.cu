/* -*-c++-*-  */
#include "cuda_stdlib.h"
#include "cutil.h"
#include "stdlib.h"
#include "stdio.h"

void init_gpu() {

    float* temp;
    //note: first call of cuda library initializes the card for the current thread
    CUDA_SAFE_CALL( cudaMalloc((void**) &temp, 32) );
    CUDA_SAFE_CALL( cudaFree(temp) );
}

/** a += b;
 */
__global__ void add_self(int xDim, int yDim, int internXdim, float* a, float* b)
{
    int im_x = blockIdx.x*blockDim.x + threadIdx.x;
    int im_y = blockIdx.y*blockDim.y + threadIdx.y;	
    int offs = im_y*internXdim+im_x;

    if( im_x < xDim && im_y < yDim)
	a[offs] += b[offs];
}

/** a -= b;
 */
__global__ void sub_self(int xDim, int yDim, int internXdim, float* a, float* b)
{
    int im_x = blockIdx.x*blockDim.x + threadIdx.x;
    int im_y = blockIdx.y*blockDim.y + threadIdx.y;	
    int offs = im_y*internXdim+im_x;

    if( im_x < xDim && im_y < yDim)
	a[offs] -= b[offs];
}

__global__ void scalar_mult_self(int xDim, int yDim, int internXdim, float* a, float scalar) {

    int im_x = blockIdx.x*blockDim.x + threadIdx.x;
    int im_y = blockIdx.y*blockDim.y + threadIdx.y;
    int offs =  im_y*internXdim+im_x;

    if( im_x < xDim && im_y < yDim)
	a[offs] *= scalar;
}

/** fill an image with a value on gpu
 */
__global__ void fill(int xDim, int yDim, int internXdim, float* image, float value)
{
    int im_x = blockIdx.x*blockDim.x + threadIdx.x;
    int im_y = blockIdx.y*blockDim.y + threadIdx.y;
	
    int offs = im_y*internXdim+im_x;
    if( im_x < xDim && im_y < yDim) {
	image[offs] = value;
    }
}


float2 operator*(const float& scalar, const float2& vec) {
    float2 result = vec;
    result.x *= scalar;
    result.y *= scalar;
    return result;
}

float2 operator-(const float2& v1, const float2& v2) {

    float2 result;
    result.x = v1.x - v2.x;
    result.y = v1.y - v2.y;

    return result;
}

int2 operator+(const int2& v1, const int2& v2) {

    return make_int2(v1.x+v2.x,v1.y+v2.y);
}

float2 operator+(const float2& v1, const float2& v2) {

    return make_float2(v1.x+v2.x,v1.y+v2.y);
}

void operator+=(float2& v1, const float2& v2) {

    v1.x += v2.x;
    v1.y += v2.y;
}

bool operator==(const int2& v1, const int2& v2) {
    return (v1.x == v2.x && v1.y == v2.y);
}
bool operator==(const int3& v1, const int3& v2) {
    return (v1.x == v2.x && v1.y == v2.y && v1.z == v2.z);
}
bool operator==(const int4& v1, const int4& v2) {
    return (v1.x == v2.x && v1.y == v2.y && v1.z == v2.z && v1.w == v2.w);
}

bool operator!=(const int3& v1, const int3& v2) {
    return !operator==(v1,v2);
}
bool operator!=(const int4& v1, const int4& v2) {
    return !operator==(v1,v2);
}


float operator%(const float2& v1, const float2& v2) {
    return v1.x*v2.x + v1.y*v2.y;
}

float norm(int2 vec) {
    return sqrt((float) (vec.x*vec.x+vec.y*vec.y));
}
float norm(float2 vec) {
    return sqrt(vec.x*vec.x+vec.y*vec.y);
}


void pgm2ppm(const float* pgm, int xDim, int yDim, unsigned char*& ppm) {
    
    ppm = (unsigned char*) malloc(3*xDim*yDim*sizeof(unsigned char));
    
    for (int y=0; y < yDim; y++) {
	for (int x=0; x < xDim; x++) {
	    
	    ppm[y*3*xDim + 3*x] = (unsigned char) (255.0 * pgm[y*xDim+x]);
	    ppm[y*3*xDim + 3*x+1] = (unsigned char) (255.0 * pgm[y*xDim+x]);
	    ppm[y*3*xDim + 3*x+2] = (unsigned char) (255.0 * pgm[y*xDim+x]);
	}
    }
}
