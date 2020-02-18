/* -*-c++-*- */
#include <stdlib.h>
#include <stdio.h>
#include <cassert>
#include <algorithm>

#include "cuda_contour_viz.h"

void make_dense_contour(const int2* input, uint input_length, int2* dense_contour, uint* dense_length, int xDim, int yDim) {

    *dense_length = 0;

    if (input_length == 0) {
	printf("input empty\n");
	return;
    }

    int last_x = input[input_length-1].x;
    int last_y = input[input_length-1].y;
    
    //fprintf(stderr,"start pos: %d,%d\n",last_x,last_y);
    //fprintf(stderr,"allowed dimensions: %d,%d\n",xDim,yDim);

    assert( last_x >= 0 && last_x < xDim && last_y >= 0 && last_y < yDim);

    for (uint i=0; i < input_length; i++) {

	//printf("i: %d, dense_length: %d\n",i,*dense_length);

	int cur_x = input[i].x;
	int cur_y = input[i].y;

	assert( cur_x >= 0 && cur_x < xDim && cur_y >= 0 && cur_y < yDim);

	//printf("connecting (%d,%d) and (%d,%d)\n",last_x,last_y,cur_x,cur_y);

	if (cur_x == last_x) { //vertical line
	    //std::cerr << "vertical, y-diff: " << (cur_y-last_y) << std::endl;
	    int x = cur_x;
	    if (last_y < cur_y) {
		for (int y = last_y+1; y <= cur_y; y++) {
		    assert( x >= 0 && x < xDim && y >= 0 && y < yDim);
		    dense_contour[*dense_length].x = x;
		    dense_contour[*dense_length].y = y;
		    (*dense_length)++;
		}
	    }
	    else {
		for (int y = last_y-1; y >= cur_y; y--) {
		    assert( x >= 0 && x < xDim && y >= 0 && y < yDim);
		    dense_contour[*dense_length].x = x;
		    dense_contour[*dense_length].y = y;
		    (*dense_length)++;
		}
	    }
	    //std::cerr << "vert done" << std::endl;
	}
	else if (last_y == cur_y) { //horizontal line
	    //std::cerr << "horizontal" << std::endl;
	    int y = last_y;
	    if (last_x < cur_x) {
		for (int x = last_x+1; x <= cur_x; x++) {
		    assert( x >= 0 && x < xDim && y >= 0 && y < yDim);
		    dense_contour[*dense_length].x = x;
		    dense_contour[*dense_length].y = y;
		    (*dense_length)++;
		}
	    }
	    else {
		for (int x = last_x-1; x >= cur_x; x--) {
		    assert( x >= 0 && x < xDim && y >= 0 && y < yDim);
		    dense_contour[*dense_length].x = x;
		    dense_contour[*dense_length].y = y;
		    (*dense_length)++;
		}
	    }
	}
	else {
	    float m = float(last_y - cur_y) / float(last_x - cur_x);
	    float invm = 1.0/m;
	    if (fabs(m) > 1.0) {
		if (cur_y > last_y) {
		    for (int y = last_y+1; y <= cur_y; y++) {
			int x = (int)(0.5+last_x+(y-last_y)*invm);
			assert( x >= 0 && x < xDim && y >= 0 && y < yDim);
			dense_contour[*dense_length].x = x;
			dense_contour[*dense_length].y = y;
			(*dense_length)++;
		    }
		}
		else {
		    for (int y = last_y-1; y >= cur_y; y--) {
			int x = (int)(0.5+last_x+(y-last_y)*invm);
			assert( x >= 0 && x < xDim && y >= 0 && y < yDim);
			dense_contour[*dense_length].x = x;
			dense_contour[*dense_length].y = y;
			(*dense_length)++;
		    }
		}
	    }
	    else {
		if (cur_x > last_x) {
		    for (int x = last_x+1; x <= cur_x; x++) {
			int y = (int)(0.5+last_y+(x-last_x)*m);
			assert( x >= 0 && x < xDim);
			assert( y >= 0 && y < yDim);
			dense_contour[*dense_length].x = x;
			dense_contour[*dense_length].y = y;
			(*dense_length)++;
		    }
		}
		else {
		    for (int x = last_x-1; x >= cur_x; x--) {
			int y = (int)(0.5+last_y+(x-last_x)*m);
			assert( x >= 0 && x < xDim);
			assert( y >= 0 && y < yDim);
			dense_contour[*dense_length].x = x;
			dense_contour[*dense_length].y = y;
			(*dense_length)++;
		    }
		}
	    }
	}

	last_x = cur_x;
	last_y = cur_y;
    }

    //printf("set length to %d\n",*dense_length);
}


void mark_contour(int* segmentation, int xDim, int yDim, const int2* expl_contour, uint expl_contour_length, int width, bool clear) {

    uint contour_length;

    int2 explicit_contour[20000];
    make_dense_contour(expl_contour,expl_contour_length,explicit_contour,&contour_length,xDim,yDim);

    assert(contour_length > 0 && contour_length <= 20000);
    
    //printf("org length: %d, new length: %d\n",expl_contour_length,contour_length);

    if (clear) {
	for (int i=0; i < xDim*yDim; i++)
	    segmentation[i] = 0;
    }
    
    int** line_points = (int**) malloc(yDim*sizeof(int*));
    int** column_points = (int**) malloc(xDim*sizeof(int*));

    for (int i=0; i < yDim; i++)
	line_points[i] = (int*) malloc(1000*sizeof(int));
    for (int i=0; i < xDim; i++)
	column_points[i] = (int*) malloc(1000*sizeof(int));
    
    int* line_count = (int*) malloc(yDim*sizeof(int));
    int* column_count = (int*) malloc(xDim*sizeof(int));

    for (int i=0; i < yDim; i++)
	line_count[i] = 0;
    for (int i=0; i < xDim; i++)
	column_count[i] = 0;

    int last_x = explicit_contour[contour_length-1].x;
    int last_y = explicit_contour[contour_length-1].y;

    segmentation[last_y*xDim+last_x] = 1;
    
    for (uint k=0; k < contour_length; k++) {

	int cur_x = explicit_contour[k].x;
	int cur_y = explicit_contour[k].y;

	segmentation[cur_y*xDim+cur_x] = 1;
	
	//	std::cerr << "k: " << k << ", point (" << cur_x << "," << cur_y << ")" <<  std::endl;
	assert((cur_x-last_x)*(cur_x-last_x) +  (cur_y-last_y)*(cur_y-last_y) <= 2);

	if (cur_y != last_y) {

	    if (cur_y < last_y) {// moving upwards
		assert(line_count[cur_y] < 1000);
		line_points[cur_y][line_count[cur_y]] = cur_x;
		line_count[cur_y]++;
	    }
	    else {//moving downwards
		assert(line_count[last_y] < 1000);
		line_points[last_y][line_count[last_y]] = last_x;
		line_count[last_y]++;
	    }
	}
	
	if (cur_x != last_x) {
	    
	    if (cur_x < last_x) {// moving left
		assert(column_count[cur_x] < 1000);
		column_points[cur_x][column_count[cur_x]] = cur_y;
		column_count[cur_x]++;
	    }
	    else {
		assert(column_count[last_x] < 1000);
		column_points[last_x][column_count[last_x]] = last_y;
		column_count[last_x]++;
	    }
	}
	
	last_x = cur_x;
	last_y = cur_y;
    }

    //    fprintf(stderr,"width: %d\n",width);

    /***** draw rows ****/
    for (int y=0; y < yDim; y++) {
	assert(line_count[y] % 2 == 0);

	std::sort(line_points[y],line_points[y]+line_count[y]);

	for (size_t i=0; i < line_count[y]; i+= 2) {

	    int first_x = line_points[y][i];
	    int second_x = line_points[y][i+1];

	    for (int x=max(0,first_x-width+1); x <= first_x; x++)
		segmentation[y*xDim+x] = 1;

	    //for (int x= std::max(second_x-1,first_x); x <= second_x; x++)
	    for (int x=second_x; x <= min(second_x+width-1,xDim-1); x++) 
		segmentation[y*xDim+x] = 1;
	}
    }

    /***** draw columns ****/
    for (int x=0; x < xDim; x++) {
	assert(column_count[x] % 2 == 0);

	std::sort(column_points[x],column_points[x]+column_count[x]);

	for (size_t i=0; i < column_count[x]; i+=2) {

	    int first_y = column_points[x][i];
	    int second_y = column_points[x][i+1];

	    //for (int y=first_y; y <= std::min(first_y+1,second_y); y++) 
	    for (int y=max(0,first_y-width+1); y <= first_y; y++)
		segmentation[y*xDim+x] = 1;

	    //for (int y=std::max(second_y-1,first_y); y <= second_y; y++) 
	    for (int y=second_y; y <= min(second_y+width-1,yDim-1); y++)
		segmentation[y*xDim+x] = 1;
	}
    }


    free(line_count);
    free(column_count);

    for (int i=0; i < yDim; i++)
	free(line_points[i]);
    for (int i=0; i < xDim; i++)
	free(column_points[i]);
    free(line_points);
    free(column_points);
}
