/* -*-c++-*- */

#include <stdio.h>
#include <cutil.h>
#include <cassert>
#include <set>

#include "curv_types.h"
#include "cuda_stdlib.h"
#include "cuda_gradient.h"

//source-include
#include "cuda_ratio_math.cu"
#include "cuda_contour_viz.h"

texture<int,2,cudaReadModeElementType> num_tex; //contains an entry for every point and each of the 49 considered target pixles
texture<int,2,cudaReadModeElementType> denom_tex; //contains the regularity estimates for every pair of the 49 possible increments

float line_integral(const float* image, int xDim, int yDim, int zDim, int z,
		    int ox, int oy, int tx, int ty, bool grad_abs = false) {

  float2 n;
  n.x = -(ty-oy);
  n.y = (tx-ox);

  float n_norm = sqrt(n.x*n.x + n.y*n.y);
  n.x /= n_norm;
  n.y /= n_norm;
    
  //  draw line from (ox,oy) to (tx,ty), excluding (ox,oy)

  float2 grad = gradient(image,ox,oy,z,xDim,yDim,zDim);
  float result = 0.5 * ( (grad_abs) ? -norm(grad) :  grad % n);

  if (tx == ox) { //vertical line
    int x = ox;
    if (oy < ty) {
      for (int y = oy+1; y <= ty; y++) {
	grad = gradient(image,x,y,z,xDim,yDim,zDim);
	result += (grad_abs) ? -norm(grad) : grad % n;
      }
    }
    else {
      for (int y = oy-1; y >= ty; y--) {
	grad = gradient(image,x,y,z,xDim,yDim,zDim);
	result += (grad_abs) ? -norm(grad) : grad % n;
      }
    }
  }
  else if (oy == ty) { //horizontal line
    int y = oy;
    if (ox < tx) {
      for (int x = ox+1; x <= tx; x++) {
	grad = gradient(image,x,y,z,xDim,yDim,zDim);
	result += (grad_abs) ? -norm(grad) : grad % n;
      }
    }
    else {
      for (int x = ox-1; x >= tx; x--) {
	grad = gradient(image,x,y,z,xDim,yDim,zDim);
	result += (grad_abs) ? -norm(grad) : grad % n;
      }
    }
  }
  else {
    float m = float(oy - ty) / float(ox - tx);
    float invm = 1.0/m;
    if (fabs(m) > 1.0) {
      if (ty > oy) {
	for (int y = oy+1; y <= ty; y++) {
	  int x = (int)(0.5+ox+(y-oy)*invm);
	  grad = gradient(image,x,y,z,xDim,yDim,zDim);
	  result += (grad_abs) ? -norm(grad) : grad % n;
	}
      }
      else {
	for (int y = oy-1; y >= ty; y--) {
	  int x = (int)(0.5+ox+(y-oy)*invm);
	  grad = gradient(image,x,y,z,xDim,yDim,zDim);
	  result += (grad_abs) ? -norm(grad) : grad % n;
	}
      }
    }
    else {
      if (tx > ox) {
	for (int x = ox+1; x <= tx; x++) {
	  int y = (int)(0.5+oy+(x-ox)*m);
	  grad = gradient(image,x,y,z,xDim,yDim,zDim);
	  result += (grad_abs) ? -norm(grad) : grad % n;
	}
      }
      else {
	for (int x = ox-1; x >= tx; x--) {
	  int y = (int)(0.5+oy+(x-ox)*m);
	  grad = gradient(image,x,y,z,xDim,yDim,zDim);
	  result += (grad_abs) ? -norm(grad) : grad % n;
	}
      }
    }
  }

  grad = gradient(image,tx,ty,z,xDim,yDim,zDim);
  result -= 0.5 * ( (grad_abs) ? -norm(grad) :  grad % n);

  return result;
}

__global__ void calc_refined_distance(const int* old_distance_num, const int* old_distance_denom,
				      const int xDim, const int yDim, const int ratio_num, const int ratio_denom,
				      const int* denom_weights, 
				      int* dev_status, int* new_distance_num, int* new_distance_denom, int* trace,
				      const int* old_changed, int* new_changed) {
    
  int im_x = blockIdx.x * blockDim.x + threadIdx.x;
  int im_y = blockIdx.y * blockDim.y + threadIdx.y;
    
  int changes = 0;
    
  if (im_x < xDim && im_y < yDim) {
	
    for (int dx1 = -3; dx1 <= 3; dx1++) {
      for (int dy1 = -3; dy1 <= 3; dy1++) {
		
	if (dx1 != 0 || dy1 != 0) {

	  int base = (dy1+3)*7 + (dx1+3);

	  int addr = (im_y + base*yDim)*xDim + im_x;

	  int new_dist_num = old_distance_num[addr];
	  int new_dist_denom = old_distance_denom[addr];
	  int trace_entry = trace[addr];

	  int num_int = tex2D(num_tex,im_x,(im_y + base*yDim));

	  int ox = im_x - dx1;
	  int oy = im_y - dy1;

	  //int nBetter = 0;

	  if (ox >= 0 && oy >= 0 && ox < xDim && oy < yDim 
	      && old_changed[oy*xDim+ox]) {

	    base *= 49; //!!!

	    for (int d2=0; d2 < 49; d2++) {

	      int denom_int = denom_weights[base + d2];
				
	      if (denom_int <= 100000) {
		int old_num   = old_distance_num[(oy+d2*yDim)*xDim+ox];
		int old_denom = old_distance_denom[(oy+d2*yDim)*xDim+ox];
		int hyp_num   = old_num + num_int;
		int hyp_denom = old_denom + denom_int;
	    
		if (dist_lower(new_dist_num,new_dist_denom,hyp_num,hyp_denom,ratio_num,ratio_denom,dev_status)) {

		  //nBetter++;

		  //DEBUG
		  // if (im_x==308 && im_y == 5 && dx1==3 && dy1==-2) {
		  //   dev_debug[0] = nBetter;
		  //   dev_debug[4*(nBetter-1) + 1] = new_dist_num;
		  //   dev_debug[4*(nBetter-1) + 2] = new_dist_denom;
		  //   dev_debug[4*(nBetter-1) + 3] = hyp_num;
		  //   dev_debug[4*(nBetter-1) + 4] = hyp_denom;
		  // }
		  //END_DEBUG

		  new_dist_num = hyp_num;
		  new_dist_denom = hyp_denom;
		  trace_entry = d2;
				    
		  changes = 1; //mark that distance changed
		}
	      }
	    }
	  }	

	  new_distance_num[addr] = new_dist_num;
	  new_distance_denom[addr] = new_dist_denom;
	  trace[addr] = trace_entry;
	}
      }
    }

    if (changes == 1) {
      dev_status[0] = 1;
    }
    new_changed[im_y*xDim+im_x] = changes;
  }
}

int4 visit(int x, int y, int dx, int dy, int time, uint* time_stamp, int xDim, int yDim, int* trace) {

  int base = (dy+3)*7 + (dx+3);
  int addr = (y + base*yDim)*xDim + x;

  if (time_stamp[addr] < time)
    return make_int4(-1,-1,-1,-1);
  else if (time_stamp[addr] == time)
    return make_int4(x,y,dx,dy);
  else {
    time_stamp[addr] = time;
    if (trace[addr] != -1) {
      
      assert(trace[addr] >= 0 && trace[addr] < 49);

      int ox = x-dx;
      int oy = y-dy;

      int dx_new = (trace[addr] % 7) -3;
      int dy_new = (trace[addr] / 7) -3;

      assert(ox >= 0 && ox < xDim);
      assert(oy >= 0 && oy < yDim);
    
      return visit(ox,oy,dx_new,dy_new,time,time_stamp,xDim,yDim,trace );
    }
    else
      return make_int4(-1,-1,-1,-1);
  }

}

void update_ratio(int4 trace_start, int* trace, int* host_num, int* host_denom_weight, int xDim, int yDim, 
		  int& ratio_num, int& ratio_denom, int2* alignment, int* alignment_length, int max_length
		  /*DEBUG int* host_dist_num, int* host_dist_denom, int* prev_host_dist_num, int* prev_host_dist_denom END_DEBUG*/) {


  bool* traced = (bool*) malloc(xDim*yDim*49*sizeof(bool));
  for (int i=0; i < xDim*yDim*49; i++)
    traced[i] = false;

  int4 cur_node = trace_start;
  while(!traced[(cur_node.y + ((cur_node.w+3)*7 + (cur_node.z+3))*yDim  )*xDim+cur_node.x]) {
    traced[(cur_node.y + ((cur_node.w+3)*7 + (cur_node.z+3))*yDim  )*xDim+cur_node.x] = true;

    //fprintf(stderr,"visiting pos %d,%d\n",cur_node.x,cur_node.y);

    int ox = cur_node.x - cur_node.z;
    int oy = cur_node.y - cur_node.w;

    int trace_entry = trace[(cur_node.y + ((cur_node.w+3)*7 + (cur_node.z+3))*yDim )*xDim+cur_node.x];

    assert(trace_entry >= 0 && trace_entry < 49);

    int dx = (trace_entry % 7) - 3;
    int dy = (trace_entry / 7) - 3;
	
    cur_node.x = ox;
    cur_node.y = oy;
    cur_node.z = dx;
    cur_node.w = dy;
  }

  int4 root_node = cur_node;

  int num_sum = 0;
  int denom_sum = 0;

  *alignment_length = 0;

  do {
    assert((*alignment_length) < max_length);
    alignment[*alignment_length].x = cur_node.x;
    alignment[*alignment_length].y = cur_node.y;
    (*alignment_length)++;

    num_sum  += host_num[(cur_node.y+ ((cur_node.w+3)*7 + (cur_node.z+3))*yDim )*xDim+cur_node.x];
    //	printf("num_sum: %d\n",num_sum);

    int trace_entry = trace[(cur_node.y+ ((cur_node.w+3)*7 + (cur_node.z+3))*yDim )*xDim+cur_node.x];
    assert(trace_entry >= 0 && trace_entry < 49);

    int next_dx = (trace_entry % 7) - 3;
    int next_dy = (trace_entry / 7) - 3;

    int ox = cur_node.x - cur_node.z;
    int oy = cur_node.y - cur_node.w;

    denom_sum += host_denom_weight[(cur_node.w+3)*343 + (cur_node.z+3)*49 + (next_dy+3)*7 + (next_dx+3)];

    cur_node.x = ox;
    cur_node.y = oy;
    cur_node.z = next_dx;
    cur_node.w = next_dy;

  } while (cur_node != root_node);
    
  free(traced);    
    
  fprintf(stderr,"found cycle of length %d with ratio %f\n",*alignment_length,((double) num_sum)/((double) denom_sum));

  fprintf(stderr,"old ratio: %d / %d\n",ratio_num,ratio_denom);
  fprintf(stderr,"new ratio: %d / %d\n",num_sum,denom_sum);
  assert( ((double) num_sum)/((double) denom_sum) < ((double) ratio_num)/((double) ratio_denom) );

  ratio_num = num_sum;
  ratio_denom = denom_sum;
}


extern void find_refined_curv_ratio_cycle(const float* image, int xDim, int yDim, int zDim,
					  float length_weight, float curv_power,
					  int2* alignment, int* alignment_length, int max_length,
					  CurvEstimatorType curv_type, bool grad_abs) {

  const float pi = 3.1415926535897931f;

  int* host_num = (int*) malloc(xDim*yDim*49*sizeof(int));

  fprintf(stderr,"image of size %d x %d\n",xDim,yDim);

  /*** compute data term as discretized line integral ***/
  for (int y = 0; y < yDim; y++) {
    for (int x = 0; x < xDim; x++) {

      for (int dx = -3; dx <= 3; dx++) {
	for (int dy = -3; dy <= 3; dy++) {

	  int ox = x-dx;
	  int oy = y-dy;

	  int addr = (((dy+3)*7 + (dx+3))*yDim + y)*xDim + x;

	  if (ox >= 0 && ox < xDim && oy >= 0 && oy < yDim
	      && ((ox != x)  || (oy != y))) {

	    float weight = 0.0;
	    for (int z=0; z < zDim; z++) {
	      weight += line_integral(image, xDim, yDim, zDim, z, ox, oy, x, y, grad_abs);
	    }

	    host_num[addr] = roundf(1000.0 * weight / ((float) zDim));
	  }
	  else 
	    host_num[addr] = 1000000;
	}
      }

    }
  }

  int* host_denom_weights = (int*) malloc(49*49*sizeof(int));

  printf("%d\n",curv_type);

  for (int dx1 = -3; dx1 <= 3; dx1++) {
    for (int dy1 = -3; dy1 <= 3; dy1++) {

      float phi1 = 8.0f * atan2((float)dy1,(float)dx1) / (2.0f*pi);

      for (int dx2 = -3; dx2 <= 3; dx2++) {
	for (int dy2 = -3; dy2 <= 3; dy2++) {

	  int addr = (dy1+3)*343 + (dx1+3)*49 + (dy2+3)*7 + (dx2+3);

	  if ( (dx1 == 0 && dy1 == 0) || (dx2 == 0 && dy2 == 0) )
	    host_denom_weights[addr] = 1000000;
	  else {
			
	    float phi2 = 8.0f * atan2((float)dy2,(float)dx2) / (2.0f*pi);
			
	    int addr = (dy1+3)*343 + (dx1+3)*49 + (dy2+3)*7 + (dx2+3);
	    float len1 = sqrt(((float)(dx1*dx1+dy1*dy1)));
	    float len2 = sqrt(((float)(dx2*dx2+dy2*dy2)));

	    float diff = fabs(phi1 - phi2);
	    diff = std::min(diff,8.0f - diff);

	    switch(curv_type) {
	    case CHAS_SIMPLE: {
			    
	      if ((diff >= 3.0f)) {
		host_denom_weights[addr] = 1000000;
		printf("(%d,%d) -> (%d,%d) curvature: %f\n",dx1,dy1,dx2,dy2,100000.0);
	      }
	      else {
		float len = 0.5f * ( len1 + len2 );
		float f = len * pow(diff / len, curv_power) + len*length_weight;
		printf("(%d,%d) -> (%d,%d) curvature: %f\n",dx1,dy1,dx2,dy2,diff/len);

		host_denom_weights[addr] = (int) roundf(1000.0 * f);
	      }
	      break;
	    }
	    case CHAS_REFINED: {
	      if ((diff >= 3.0f)) {
		host_denom_weights[addr] = 1000000;
		//printf("(%d,%d) -> (%d,%d) curvature: %f\n",dx1,dy1,dx2,dy2,100000.0);
	      }
	      else {

		int tdx = dx1+dx2;
		int tdy = dy1+dy2;
		float len3 = sqrt((float) (tdx*tdx+tdy*tdy));
		if (len3 == 0.0f)
		  host_denom_weights[addr] = 1000000;
		else {
		  float area = sqrt( std::max(0.0f,(len2+len3)*(len2+len3) - len1*len1));
		  area *= sqrt( std::max(0.0f,len1*len1 - (len2-len3)*(len2-len3)));
		  float abs_curv =  1.414 * area / (len1*len2*len3);
				    
		  if (diff == 2.0f && abs(dx1) == 1.0 && abs(dx2) == 1.0 && abs(dy1) == 1.0 && abs(dy2) == 1.0)
		    printf("(%d,%d) -> (%d,%d) curvature: %f\n",dx1,dy1,dx2,dy2,abs_curv);
				
		  float len = 0.5f * (len1 + len2); 
				    
		  float f = len * pow(abs_curv, curv_power) + len*length_weight;
		  host_denom_weights[addr] = (int) roundf(1000.0 * f);
		  assert(host_denom_weights[addr] >= 0);
		}
	      }

	      break;
	    }
	    case DC: {
	      assert(phi1 >= -8.0);
	      assert(phi2 >= -8.0);
	      float alpha = (4.0-diff) * (2.0f*pi / 8.0);
	      float tan_num = (len1 / len2) - cos(alpha);
	      float tan_denom = sin(alpha);

	      float alpha1 = atan2(tan_num,tan_denom);
	      float abs_curv = (4.0 / 3.14159) * (2.0 * cos(alpha1)) / len2;
	      if (alpha < 0.5*pi)
		abs_curv += 4.0;
	      float len = 0.5f * (len1 + len2); 

	      //printf("(%d,%d) -> (%d,%d) curvature: %f\n",dx1,dy1,dx2,dy2,abs_curv);

	      float f = len * pow(abs_curv, curv_power) + len*length_weight;
	      host_denom_weights[addr] = (int) roundf(1000.0 * f);
	      assert(host_denom_weights[addr] >= 0);

	      break;
	    }
	    case BRUCK: {
	      diff *= 2.0f*pi / 8.0f;
	      float curv_weight = pow(diff,curv_power) / pow(0.5f*min(len1,len2),curv_power-1.0f);
	      float len = 0.5f * (len1 + len2);
	      float f = curv_weight + len*length_weight;
	      host_denom_weights[addr] = (int) roundf(1000.0 * f);
	      assert(host_denom_weights[addr] >= 0);
	      break;
	    }
	    default: {
	      fprintf(stderr,"invalid curvature estimator type\n");
	      exit(0);
	    }
	    } //end of switch statement
	  }
	}
      }
    }
  }


  //DEBUG - check symmetry of curvature weights 
  //if (curv_type != BRUCK) {
  if (true) {
    for (int dx1 = -3; dx1 <= 3; dx1++) {
      for (int dy1 = -3; dy1 <= 3; dy1++) {
	for (int dx2 = -3; dx2 <= 3; dx2++) {
	  for (int dy2 = -3; dy2 <= 3; dy2++) {
			
	    int addr1 = (dy1+3)*343 + (dx1+3)*49 + (dy2+3)*7 + (dx2+3);
	    int addr2 = (dy2+3)*343 + (dx2+3)*49 + (dy1+3)*7 + (dx1+3);
			
	    if (!(host_denom_weights[addr1] == host_denom_weights[addr2])) {
	      fprintf(stderr,"forward: %d\n",host_denom_weights[addr1]);
	      fprintf(stderr,"backward: %d\n",host_denom_weights[addr2]);
	    }

	    assert(abs(host_denom_weights[addr1] - host_denom_weights[addr2]) <= 1);
	  }
	}
      }
    }
  }
  //END_DEBUG

  const int BLOCK_SIZE = 8;
  const dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);

  int upper_xDim = xDim;
  while (upper_xDim % BLOCK_SIZE != 0)
    upper_xDim++;
  int upper_yDim = yDim;
  while (upper_yDim % BLOCK_SIZE != 0)
    upper_yDim++;
    
  dim3 dimGrid(upper_xDim/BLOCK_SIZE,upper_yDim/BLOCK_SIZE);

  cudaArray* num_array = 0;
  CUDA_SAFE_CALL( cudaMallocArray(&num_array, &num_tex.channelDesc ,xDim,49*yDim )  );
  CUDA_SAFE_CALL( cudaMemcpy2DToArray(num_array, 0,0, host_num, xDim*sizeof(int), xDim*sizeof(int), 49*yDim, cudaMemcpyHostToDevice) );

  num_tex.addressMode[0] = cudaAddressModeClamp;
  num_tex.addressMode[1] = cudaAddressModeClamp;
  num_tex.filterMode = cudaFilterModePoint;
  num_tex.normalized = false;    // access with normal texture coordinates
  CUDA_SAFE_CALL( cudaBindTextureToArray(num_tex, num_array) );

  int* device_denom_weights = 0;
  CUDA_SAFE_CALL( cudaMalloc((void**) &device_denom_weights, 49*49*sizeof(int)) );
  CUDA_SAFE_CALL( cudaMemcpy(device_denom_weights,host_denom_weights,49*49*sizeof(int),cudaMemcpyHostToDevice) );

  int* device_dist_num1 = 0;
  int* device_dist_num2 = 0;
  int* device_dist_denom1 = 0;
  int* device_dist_denom2 = 0;
  CUDA_SAFE_CALL( cudaMalloc((void**) &device_dist_num1,   xDim*yDim*49*sizeof(int)) );
  CUDA_SAFE_CALL( cudaMalloc((void**) &device_dist_denom1, xDim*yDim*49*sizeof(int)) );
  CUDA_SAFE_CALL( cudaMalloc((void**) &device_dist_num2,   xDim*yDim*49*sizeof(int)) );
  CUDA_SAFE_CALL( cudaMalloc((void**) &device_dist_denom2, xDim*yDim*49*sizeof(int)) );
  int* device_trace = 0;
  CUDA_SAFE_CALL( cudaMalloc((void**) &device_trace, xDim*yDim*49*sizeof(int)) );

  int* device_distance_changed1 = 0;
  int* device_distance_changed2 = 0;
  CUDA_SAFE_CALL( cudaMalloc((void**) &device_distance_changed1, xDim*yDim*sizeof(int)) );
  CUDA_SAFE_CALL( cudaMalloc((void**) &device_distance_changed2, xDim*yDim*sizeof(int)) );

  int* host_trace = (int*) malloc(xDim*yDim*49*sizeof(int));
  for (int i=0; i < xDim*yDim*49; i++)
    host_trace[i] = -1;
  CUDA_SAFE_CALL( cudaMemcpy(device_trace,host_trace,xDim*yDim*49*sizeof(int),cudaMemcpyHostToDevice) );

  int host_status[2] = {0,0};

  int* device_status = 0;
  CUDA_SAFE_CALL( cudaMalloc((void**) &device_status, 2*sizeof(int)) );

  int* host_dist_num = (int*) malloc(xDim*yDim*49*sizeof(int));
  assert(host_dist_num != 0);
  int* host_dist_denom = (int*) malloc(xDim*yDim*49*sizeof(int));
  assert(host_dist_denom != 0);
  
  uint* time_stamp = (uint*) malloc(xDim*yDim*49*sizeof(int));
  assert(time_stamp != 0);

  //evolve ratio 
  int ratio_num = 0;
  int ratio_denom = 1;
  do {
    /*** init distance ***/
    for (int i=0; i < xDim*yDim*49; i++) {
      host_dist_num[i] = 0;
      host_dist_denom[i] = 0;
      host_trace[i] = -1;
    }

    CUDA_SAFE_CALL( cudaMemcpy(device_dist_num1,host_dist_num,xDim*yDim*49*sizeof(int),cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(device_dist_denom1, host_dist_denom, xDim*yDim*49*sizeof(int),cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(device_trace, host_trace, xDim*yDim*49*sizeof(int),cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemset((void*) device_distance_changed1,1,xDim*yDim*sizeof(int)) );

    cudaThreadSynchronize();

    int iter = 0;
    int4 pos = make_int4(-1,-1,-1,-1);

    do {
      iter++;
      if (iter % 5 == 0)
	fprintf(stderr,"iter %d\n",iter);

      host_status[0] = 0;
      host_status[1] = 0;

      cudaThreadSynchronize();

      CUDA_SAFE_CALL( cudaMemcpy(device_status,host_status,2*sizeof(int),cudaMemcpyHostToDevice) );

      cudaThreadSynchronize();
	    
      if (iter % 2 == 1)
	calc_refined_distance<<< dimGrid, dimBlock>>>(device_dist_num1, device_dist_denom1, xDim, yDim, ratio_num, ratio_denom, 
						      device_denom_weights,device_status, device_dist_num2, device_dist_denom2, device_trace,
						      device_distance_changed1,device_distance_changed2);
      else
	calc_refined_distance<<< dimGrid, dimBlock>>>(device_dist_num2, device_dist_denom2, xDim, yDim, ratio_num, ratio_denom, 
						      device_denom_weights,device_status, device_dist_num1, device_dist_denom1, device_trace,
						      device_distance_changed2,device_distance_changed1);

      CUT_CHECK_ERROR("Kernel execution failed");

      cudaThreadSynchronize();

      // check if kernel execution generated an error
      CUT_CHECK_ERROR("Kernel execution failed");
      CUDA_SAFE_CALL( cudaMemcpy(&(host_status[0]),device_status,2*sizeof(int),cudaMemcpyDeviceToHost) );

      //fprintf(stderr,"status: %d  %d \n",host_status[0],host_status[1],host_status[2],host_status[3]);
      assert(host_status[1] == 0);

      pos = make_int4(-1,-1,-1,-1);

      if (iter % 25 == 0) {
	fprintf(stderr,"checking for cycles\n");
	/**** checking for cycles (maybe this can be done on the gpu in the future?) ****/
	CUDA_SAFE_CALL( cudaMemcpy(host_trace, device_trace, xDim*yDim*49*sizeof(int),cudaMemcpyDeviceToHost) );

	for (int i=0; i < xDim*yDim*49; i++) 
	  time_stamp[i] = 0xFFFFFFFF;

	int time = 0;

	for (int y=0; y < yDim; y++) {
	  for (int x=0; x < xDim; x++) {
	    for (int dx = -3; dx <= 3; dx++) {
	      for (int dy = -3; dy <= 3; dy++) {

		if (dx != 0 || dy != 0) {
		  int base = (dy+3)*7 + (dx+3);
		
		  int addr = (y + base*yDim)*xDim + x;
		  
		  if (time_stamp[addr] == 0xFFFFFFFF) {
		    time++;
		    pos = visit(x,y,dx,dy,time,time_stamp,xDim,yDim,host_trace);
		    
		    if (pos.x != -1 && pos.y != -1)
		      goto cycle_check_finished;
		  }
		}
	      }
	    }
	  }
	}
		
	fprintf(stderr,"time: %d\n",time);
      cycle_check_finished:
	if (pos.x != -1 && pos.y != -1) {
	  fprintf(stderr,"cycle found\n");
	  break;
	}
      }

    } while (host_status[0] != 0 || (pos.x != -1 && pos.y != -1 && pos.z != -1));


    if (pos.x != -1 && pos.y != -1) {
	  
      fprintf(stderr,"found cycle starting at %d,%d\n",pos.x,pos.y);

      update_ratio(pos, host_trace, host_num, host_denom_weights, xDim, yDim, 
		   ratio_num, ratio_denom, alignment, alignment_length, max_length
		   /*DEBUG host_dist_num, host_dist_denom, prev_host_dist_num, prev_host_dist_denom END_DEBUG*/);

      fprintf(stderr,"cycle length: %d\n",*alignment_length);
    }

  } while(host_status[0] != 0); 


  CUDA_SAFE_CALL( cudaFree(device_status) );
  CUDA_SAFE_CALL( cudaFree(device_trace) );
  CUDA_SAFE_CALL( cudaFree(device_dist_num1) );
  CUDA_SAFE_CALL( cudaFree(device_dist_denom1) );
  CUDA_SAFE_CALL( cudaFree(device_dist_num2) );
  CUDA_SAFE_CALL( cudaFree(device_dist_denom2) );
    
  CUDA_SAFE_CALL( cudaFree(device_distance_changed1) );
  CUDA_SAFE_CALL( cudaFree(device_distance_changed2) );

  CUDA_SAFE_CALL( cudaFreeArray(num_array) );
  CUDA_SAFE_CALL( cudaFree(device_denom_weights) );

  free (host_trace);
  free (host_num);
  free (host_denom_weights);
  free (host_dist_num);
  free (host_dist_denom);
}


void find_refined_curv_ratio_cycle(const float* image, int xDim, int yDim, int zDim,
 				   float length_weight, float curv_power,
				   CurvEstimatorType curv_type, bool grad_abs, float* out_image) {


  int2 alignment[60000];
  int alignment_length;

  find_refined_curv_ratio_cycle(image, xDim, yDim, zDim, length_weight,curv_power,
				alignment, &alignment_length, 60000, curv_type, grad_abs);

  if (zDim != 3) {
    for (uint k=0; k < xDim*yDim; k++) {
      
      for (uint l=0; l < 3; l++)
	out_image[3*k+l] = 0.8*image[k];
    }
  }
  else {
    for (uint k=0; k < xDim*yDim*3; k++) {
      out_image[k] = 0.8*image[k];
    }
  }

  int* segmentation = (int*) malloc(xDim*yDim*sizeof(int));
  
  int lw = 4;
  mark_contour(segmentation, xDim, yDim, alignment, alignment_length, lw);
  
  for (uint k=0; k < xDim*yDim; k++) {

    if (segmentation[k] >= 1) {
      
      out_image[3*k+0] = 255;
      out_image[3*k+1] = 255;
      out_image[3*k+2] = 120;
    }
  }

  free(segmentation);
}

