#ifndef CUDASTDLIB_H
#define CUDASTDLIB_H

//useful when measuring small time units (then the card initialization messes up results)
void init_gpu();

__global__ void fill(int xDim, int yDim, int internXdim, float* image, float value);
__global__ void add_self(int xDim, int yDim, int internXdim, float* a, float* b); //a += b
__global__ void sub_self(int xDim, int yDim, int internXdim, float* a, float* b); //a -= b
__global__ void scalar_mult_self(int xDim, int yDim, int internXdim, float* a, float scalar);

/*** operators for types defined by NVidia ***/
float2 operator*(const float& scalar, const float2& vec);
float2 operator-(const float2& v1, const float2& v2);
void operator+=(float2& v1, const float2& v2);
int2 operator+(const int2& v1, const int2& v2);
float2 operator+(const float2& v1, const float2& v2);
bool operator==(const int2& v1, const int2& v2);
bool operator==(const int3& v1, const int3& v2);
bool operator==(const int4& v1, const int4& v2);

bool operator!=(const int3& v1, const int3& v2);
bool operator!=(const int4& v1, const int4& v2);

float operator%(const float2& v1, const float2& v2);

float norm(int2 vec);
float norm(float2 vec);

void pgm2ppm(const float* pgm, int xDim, int yDim, unsigned char*& ppm);


#endif
