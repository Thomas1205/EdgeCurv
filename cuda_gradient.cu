/* -*-c++-*- */

#include "cuda_gradient.h"

float image_val(const float* image, float x, float y, int xDim, int yDim) {

    if (x < 0.0f)
	x = 0.0f;
    if (y < 0.0f)
	y = 0.0f;

    if ( x > ((float) (xDim-1)))
	x=xDim -1;
    if (y > ((float) (yDim-1)))
	y=yDim -1;
	    
    
    int floor_x = ((int) floor(x));
    int ceil_x = ((int) ceil(x));
    
    int floor_y = ((int) floor(y));
    int ceil_y = ((int) ceil(y));

    float alpha_x = 1.0f - (x - ((float) floor_x));
    float alpha_y = 1.0f - (y - ((float) floor_y));

/*     assert(!isnan(image(floor_x,floor_y))); */
/*     assert(!isnan(image(floor_x,ceil_y))); */
/*     assert(!isnan(image(ceil_x,floor_y))); */
/*     assert(!isnan(image(ceil_x,ceil_y))); */

    float upper = alpha_x * ((float) image[floor_y*xDim+floor_x]) + (1.0f - alpha_x) * ((float) image[floor_y*xDim+ceil_x]);
    float lower = alpha_x * ((float) image[ceil_y*xDim+floor_x]) + (1.0f - alpha_x) * ((float) image[ceil_y*xDim+ceil_x]);

    return alpha_y * upper + (1.0f - alpha_y) * lower;
}


float2 gradient(const float* image, int x, int y, int xDim, int yDim) {

    float2 result;

    float here = image[y*xDim+x];
    float left = (x > 0) ? image[y*xDim+x-1] : here;
    float right = (x+1 < xDim) ? image[y*xDim+x+1] : here;

    result.x = 0.5f * (right - left);
    
    float up = (y > 0) ? image[(y-1)*xDim+x] : here;
    float down = (y+1 < yDim) ? image[(y+1)*xDim+x] : here;

    result.y = 0.5f * (down - up);

    return result;
}

float2 gradient(const float* image, int x, int y, int z, int xDim, int yDim, int zDim) {

    float2 result;

    float here = image[y*zDim*xDim+x*zDim+z];
    float left = (x > 0) ? image[y*zDim*xDim+(x-1)*zDim+z] : here;
    float right = (x+1 < xDim) ? image[y*zDim*xDim+(x+1)*zDim+z] : here;

    result.x = 0.5f * (right - left);
    
    float up = (y > 0) ? image[(y-1)*zDim*xDim+x*zDim+z] : here;
    float down = (y+1 < yDim) ? image[(y+1)*zDim*xDim+x*zDim+z] : here;

    result.y = 0.5f * (down - up);

    return result;
}

float2 subpixel_gradient(const float* image, float x, float y, int xDim, int yDim) {

    float2 result;

    float left = image_val(image,x-1,y,xDim,yDim);
    float right =  image_val(image,x+1,y,xDim,yDim);

    result.x = 0.5f * (right - left);
    
    float up = image_val(image,x,y-1,xDim,yDim);
    float down = image_val(image,x,y+1,xDim,yDim);

    result.y = 0.5f * (down - up);

    return result;

}
