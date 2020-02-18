/*** written by Thomas Schoenemann as an employee of Lund University, Sweden, August 2010 ***/


// implementation of minimum ratio cycles, currently based on (integer) linear programming
// TODO: write graph-based routine

#include "application.hh"
#include "grayimage.hh"
#include "colorimage.hh"
#include "conversion.hh"
#include "draw_segmentation.hh"
#include "ratio_segmentation.hh"
#include "timing.hh"

#ifdef USE_CUDA
#include "cuda_refined_curv_ratio.cuh"
#endif

int main(int argc, char** argv)
{

  if (argc == 1 || (argc == 2 && strings_equal(argv[1],"-h"))) {

    std::cerr << "USAGE: " << argv[0] << std::endl
              << "  -i <pgm or ppm> : filename of input image (to be segmented)" << std::endl
              << "  -lambda <double> : length weight" << std::endl
              << "  -o <filename> : name of the output segmentation" << std::endl
              << " [-n (4|8|16)]: size of neighborhood, default 8" << std::endl;

    exit(0);
  }

  const int nParams = 7;
  ParamDescr  params[nParams] = {{"-i",mandInFilename,0,""},{"-lambda",optWithValue,1,"1.0"},
    {"-o",mandOutFilename,0,""},{"-n",optWithValue,1,"8"},
    {"-bruckstein",flag,0,""},{"-cuda",flag,0,""},{"-curv-power",optWithValue,1,"2.0"}
  };

  Application app(argc,argv,params,nParams);

  Math3D::NamedColorImage<float> image(app.getParam("-i"),MAKENAME(image));

  const uint xDim = image.xDim();
  const uint yDim = image.yDim();

  Math3D::NamedColorImage<float> out_image(xDim,yDim,3,0.0,MAKENAME(out_image));

  double lambda = convert<double>(app.getParam("-lambda"));
  double curv_power = convert<double>(app.getParam("-curv-power"));
  uint neighborhood = convert<uint>(app.getParam("-n"));

  bool use_cuda = false;
#ifdef USE_CUDA
  use_cuda = app.is_set("-cuda");
#endif

  if (use_cuda) {
#ifdef USE_CUDA
    find_refined_curv_ratio_cycle(image.direct_access(), xDim, yDim, image.zDim(),
                                  lambda, curv_power, BRUCK, false, out_image.direct_access());
#endif
  }
  else
    mr_lp_segment(image, lambda, curv_power, neighborhood, out_image, app.is_set("-bruckstein"));

  out_image.savePPM(app.getParam("-o"));
}
