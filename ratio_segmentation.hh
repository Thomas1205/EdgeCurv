/*** written by Thomas Schoenemann as an employee of Lund University, Sweden, August 2010 ***/

#ifndef RATIO_SEGMENTATION_HH
#define RATIO_SEGMENTATION_HH

#include "colorimage.hh"

void mr_lp_segment(const Math3D::ColorImage<float>& input, double lambda, double curv_power, uint neighborhood,
                   Math3D::ColorImage<float>& output, bool bruckstein);


#endif
