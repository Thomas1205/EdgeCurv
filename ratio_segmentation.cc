/*** written by Thomas Schoenemann as an employee of Lund University, Sweden, August 2010 ***/

#include "ratio_segmentation.hh"

#include "sparse_matrix_description.hh"
#include "ClpSimplex.hpp"
#include "graph.h"

#include "CbcModel.hpp"
#include "OsiClpSolverInterface.hpp"

#include "CglGomory/CglGomory.hpp"
#include "CglProbing/CglProbing.hpp"
#include "CglRedSplit/CglRedSplit.hpp"
#include "CglTwomir/CglTwomir.hpp"
#include "CglMixedIntegerRounding/CglMixedIntegerRounding.hpp"
#include "CglMixedIntegerRounding2/CglMixedIntegerRounding2.hpp"
#include "CglOddHole/CglOddHole.hpp"
#include "CglLandP/CglLandP.hpp"
#include "CglClique/CglClique.hpp"
#include "CglStored.hpp"

#include "CbcHeuristic.hpp"
#include "CbcBranchActual.hpp"

#include "timing.hh"

#include "vector.hh"
#include "matrix.hh"
#include "curvature.hh"
#include "gradient.hh"
#include "line_drawing.hh"
#include "grid.hh"

#include <algorithm>

#ifdef HAS_GUROBI
#define USE_GUROBI
#endif

//#define USE_CPLEX
//#define USE_XPRESS

#ifdef USE_GUROBI
#include "gurobi_c++.h"
#endif

#ifdef USE_CPLEX
#include <ilcplex/cplex.h>
#endif

#ifdef USE_XPRESS
#include "xprs.h"
#endif


/***********************************************************************************************/

void mr_lp_segment(const Math3D::ColorImage<float>& input, double lambda, double curv_power, uint neighborhood,
                   Math3D::ColorImage<float>& output, bool bruckstein)
{

  bool node_constraints = true;
  bool edge_constraints = true;

  const uint xDim = input.xDim();
  const uint yDim = input.yDim();
  const uint zDim = input.zDim();

  Storage1D<Math3D::NamedTensor<float> > gradient(zDim,MAKENAME(gradient));
  for (uint z=0; z < zDim; z++)
    gradient[z].resize(xDim,yDim,2);

  compute_channel_gradients(input, gradient);


  /******* generate grid ********/

  Grid grid(xDim+1,yDim+1);

  for (uint y=0; y <= yDim; y++) {
    for (uint x=0; x <= xDim; x++) {

      if (x > 0)
        grid.add_line(x,y,x-1,y);
      if (x+1 <= xDim)
        grid.add_line(x,y,x+1,y);
      if (y > 0)
        grid.add_line(x,y,x,y-1);
      if (y+1 <= yDim)
        grid.add_line(x,y,x,y+1);

      if (neighborhood >= 8) {

        if (x > 0 && y > 0)
          grid.add_line(x,y,x-1,y-1);
        if (x > 0 && y+1 <= yDim)
          grid.add_line(x,y,x-1,y+1);
        if (x+1 <= xDim && y > 0)
          grid.add_line(x,y,x+1,y-1);
        if (x+1 <= xDim && y+1 <= yDim)
          grid.add_line(x,y,x+1,y+1);
      }
      if (neighborhood >= 16) {

        if (x > 1 && y > 0)
          grid.add_line(x,y,x-2,y-1);
        if (x > 1 && y+1 < yDim)
          grid.add_line(x,y,x-2,y+1);

        if (x+2 < xDim && y > 0)
          grid.add_line(x,y,x+2,y-1);
        if (x+2 < xDim && y+1 < yDim)
          grid.add_line(x,y,x+2,y+1);

        if (x > 0 && y > 1)
          grid.add_line(x,y,x-1,y-2);
        if (x > 0 && y+2 < yDim)
          grid.add_line(x,y,x-1,y+2);

        if (x+1 < xDim && y > 1)
          grid.add_line(x,y,x+1,y-2);
        if (x+1 < xDim && y+2 < yDim)
          grid.add_line(x,y,x+1,y+2);
      }
    }
  }

  uint nLines = grid.nLines();

  grid.generate_grid_line_pairs();

  uint nLinePairs = grid.nLinePairs();

  /******* compute data terms for each line ********/

  Math1D::NamedVector<float> line_data_term(nLines,0.0,MAKENAME(line_data_term));

  for (uint l=0; l < nLines; l++) {

    const GridLine& cur_line = grid.get_line(l);

    float dx = ((int) cur_line.x2_) - ((int) cur_line.x1_);
    float dy = ((int) cur_line.y2_) - ((int) cur_line.y1_);

    //std::cerr << "dx: " << dx << std::endl;
    //std::cerr << "dy: " << dy << std::endl;

    float edge_length = sqrt(dx*dx+dy*dy);

    float nx = -dy;
    float ny = dx;

    float cost = 0.0;

    if (fabs(dx) == 1.0 && fabs(dy) == 1.0) {
      //diagonal line -> take the gradient of the pixel center

      const uint x = std::min(cur_line.x2_,cur_line.x1_);
      const uint y = std::min(cur_line.y2_,cur_line.y1_);

      cost += gradient[0](x,y,0)*nx + gradient[0](x,y,1)*ny;
    }
    else if (dx == 0.0) {

      const uint y = std::min(cur_line.y2_,cur_line.y1_);
      uint x = cur_line.x1_;
      if (x == xDim)
        x--;
      else if (x > 0 && gradient[0].norm(x-1,y) >= gradient[0].norm(x,y))
        x--;

      cost += gradient[0](x,y,0)*nx + gradient[0](x,y,1)*ny;
    }
    else if (dy == 0.0) {

      const uint x = std::min(cur_line.x2_,cur_line.x1_);
      uint y = cur_line.y1_;
      if (y == yDim)
        y--;
      else if (y > 0 && gradient[0].norm(x,y-1) >= gradient[0].norm(x,y))
        y--;

      cost += gradient[0](x,y,0)*nx + gradient[0](x,y,1)*ny;
    }
    else if (fabs(dx) == 2) {

      assert(fabs(dy) == 1);

      uint y = std::min(cur_line.y2_,cur_line.y1_);
      uint x = std::min(cur_line.x2_,cur_line.x1_);

      cost += 0.5 * (gradient[0](x,y,0)*nx + gradient[0](x,y,1)*ny
                     + gradient[0](x+1,y,0)*nx + gradient[0](x+1,y,1)*ny);
    }
    else if (fabs(dy) == 2) {

      assert(fabs(dx) == 1);

      uint y = std::min(cur_line.y2_,cur_line.y1_);
      uint x = std::min(cur_line.x2_,cur_line.x1_);

      cost += 0.5 * (gradient[0](x,y,0)*nx + gradient[0](x,y,1)*ny
                     + gradient[0](x,y+1,0)*nx + gradient[0](x,y+1,1)*ny);
    }
    else {
      TODO("32-connectivity");
    }

    cost *= edge_length;

    line_data_term[l] = cost;
  }

  /******* determine numerator and denominator cost **********/

  Math1D::Vector<float> numerator_cost(nLinePairs,0.0);
  Math1D::Vector<float> denominator_cost(nLinePairs,0.0);

  for (uint k=0; k < nLinePairs; k++) {

    const GridLinePair& pair = grid.get_line_pair(k);

    uint l1 = pair.line1_;
    uint l2 = pair.line2_;

    numerator_cost[k] = 0.5 * (line_data_term[l1] + line_data_term[l2]);

    const GridLine& line1 = grid.get_line(l1);
    const GridLine& line2 = grid.get_line(l2);

    double x1 = line1.x1_;
    double y1 = line1.y1_;
    double x2 = line1.x2_;
    double y2 = line1.y2_;
    double x3 = line2.x2_;
    double y3 = line2.y2_;

    assert(x2 == line2.x1_);
    assert(y2 == line2.y1_);

    double cw = curv_weight(x1, y1, x2, y2, x3, y3, curv_power, bruckstein);

    //std::cerr << "cw: " << cw << std::endl;


    denominator_cost[k] = cw + 0.5 * lambda * (line1.length() + line2.length());
  }

  std::cerr << "max denom-weight: " << denominator_cost.max() << std::endl;


  /******* setup the constraint system ********/

  uint nVars = nLinePairs;
  uint nConstraints = nLines; // flow conservation

  const uint node_con_offs = nConstraints;

  if (node_constraints)
    nConstraints += (xDim+1)*(yDim+1);

  const uint edge_intersec_con_offs = nConstraints;

  if (edge_constraints) {
    if (neighborhood >= 8)
      nConstraints += xDim*yDim;
    //TODO: 16-connect.
  }


  Math1D::NamedVector<double> var_lb(nVars,0.0,MAKENAME(var_lb));
  Math1D::NamedVector<double> var_ub(nVars,1.0,MAKENAME(var_ub));

  Math1D::NamedVector<double> cost(nVars,0.0,MAKENAME(cost)); //to be recomputed in every iteration

  Math1D::NamedVector<double> rhs_lower(nConstraints,0.0,MAKENAME(rhs_lower));
  Math1D::NamedVector<double> rhs_upper(nConstraints,0.0,MAKENAME(rhs_upper));

  uint nEntries = 2*nLinePairs; // for the flow conservation

  if (node_constraints)
    nEntries += nLinePairs;

  if (edge_constraints) {
    if (neighborhood >= 8)
      nEntries += 2*nLinePairs;
    //TODO: 16-connect
  }

  SparseMatrixDescription<double> lp_descr(nEntries, nConstraints, nVars);


  /*** code flow conservation ***/
  //   for (uint y=0; y <= yDim; y++) {
  //     for (uint x=0; x <= xDim; x++) {

  //       uint row = y*(xDim+1)+x;

  //       std::vector<uint> indices;

  //       grid.list_incoming_line_pairs(x, y, indices);
  //       for (uint k=0; k < indices.size(); k++)
  // 	lp_descr.add_entry(row,indices[k], 1.0);

  //       grid.list_outgoing_line_pairs(x, y, indices);
  //       for (uint k=0; k < indices.size(); k++)
  // 	lp_descr.add_entry(row,indices[k], -1.0);

  //     }
  //   }

  for (uint l=0; l < nLines; l++) {

    const GridLine& cur_line = grid.get_line(l);

    const std::vector<uint>& inpairs = cur_line.ending_line_pairs_;

    for (uint k=0; k < inpairs.size(); k++)
      lp_descr.add_entry(l,inpairs[k], 1.0);

    const std::vector<uint>& outpairs = cur_line.starting_line_pairs_;

    for (uint k=0; k < outpairs.size(); k++)
      lp_descr.add_entry(l,outpairs[k], -1.0);
  }

  if (node_constraints) {

    for (uint y = 0; y <= yDim; y++) {
      for (uint x = 0; x <= xDim; x++) {

        const uint row = node_con_offs + y*(xDim+1)+x;

        rhs_lower[row] = 0.0;
        rhs_upper[row] = 1.0;

        const GridNode& cur_node = grid.get_node(x,y);

        for (uint l_in = 0; l_in < cur_node.incoming_lines_.size(); l_in++) {

          const GridLine& cur_line = grid.get_line(cur_node.incoming_lines_[l_in]);

          for (uint p=0; p < cur_line.ending_line_pairs_.size(); p++) {

            lp_descr.add_entry(row, cur_line.ending_line_pairs_[p], 1.0);
          }
        }
      }
    }
  }

  if (edge_constraints) {

    if (neighborhood >= 8) {
      for (uint y=0; y < yDim; y++) {
        for (uint x=0; x < xDim; x++) {

          uint row = edge_intersec_con_offs + y*xDim+x;

          rhs_lower[row] = 0.0;
          rhs_upper[row] = 2.0;

          for (uint i=0; i < 4; i++) {

            uint cur_line_num = 0;

            if (i==0)
              cur_line_num = grid.find_line(x,y,x+1,y+1);
            else if (i == 1)
              cur_line_num = grid.find_line(x+1,y+1,x,y);
            else if (i == 2)
              cur_line_num = grid.find_line(x+1,y,x,y+1);
            else if (i == 3)
              cur_line_num = grid.find_line(x,y+1,x+1,y);

            const GridLine& cur_line = grid.get_line(cur_line_num);

            for (uint p=0; p < cur_line.starting_line_pairs_.size(); p++)
              lp_descr.add_entry(row, cur_line.starting_line_pairs_[p], 1.0);
            for (uint p=0; p < cur_line.ending_line_pairs_.size(); p++)
              lp_descr.add_entry(row, cur_line.ending_line_pairs_[p], 1.0);
          }
        }
      }
    }

  }

  Math1D::Vector<uint> row_start(nConstraints+1);
  lp_descr.sort_by_row(row_start);

  //set-up lp-solver
#ifdef USE_GUROBI

  GRBenv*   grb_env   = NULL;
  GRBmodel* grb_model = NULL;

  /* Create environment */

  int error = GRBloadenv(&grb_env,NULL);

  assert (!error && grb_env != NULL);

  /* Create an empty model */

  error = GRBnewmodel(grb_env, &grb_model, "curv-lp", 0, NULL, NULL, NULL, NULL, NULL);
  assert(!error);

  Storage1D<char> vtype(nVars,GRB_CONTINUOUS);

  error = GRBaddvars(grb_model,nVars,0,NULL,NULL,NULL,cost.direct_access(),var_lb.direct_access(),
                     var_ub.direct_access(),vtype.direct_access(),NULL);
  assert(!error);

  error = GRBupdatemodel(grb_model);
  assert(!error);

  for (uint c=0; c < nConstraints; c++) {

    //     if ((c % 250) == 0)
    //       std::cerr << "c: " << c << std::endl;

    //     std::string s = "c" + toString(c);
    //     char cstring [256];
    //     for (uint i=0; i < s.size(); i++)
    //       cstring[i] = s[i];
    //     cstring[s.size()] = 0;

    if (rhs_lower[c] == rhs_upper[c]) {
      error = GRBaddconstr(grb_model, row_start[c+1]-row_start[c], ((int*) lp_descr.col_indices()) + row_start[c],
                           lp_descr.value() + row_start[c], GRB_EQUAL, rhs_lower[c], NULL);
    }
    else {

      error = GRBaddrangeconstr(grb_model, row_start[c+1]-row_start[c], ((int*) lp_descr.col_indices()) + row_start[c],
                                lp_descr.value() + row_start[c], rhs_lower[c], rhs_upper[c], NULL);
    }

    assert(!error);
  }

  /* Optimize model */
  error = GRBoptimize(grb_model);
  assert(!error);

#else

#ifdef USE_CPLEX

  CPXENVptr     env = NULL;
  CPXLPptr      lp = NULL;
  int status = 0;

  /* Initialize the CPLEX environment */

  env = CPXopenCPLEX (&status);
  //CPXsetintparam(env, CPX_PARAM_STARTALG, CPX_ALG_BARRIER);
  //CPXsetintparam(env, CPX_PARAM_MIPDISPLAY, 4);
  //CPXsetintparam(env, CPX_PARAM_PREIND, CPX_OFF);
  //CPXsetintparam(env, CPX_PARAM_PREPASS, 0);

  /* If an error occurs, the status value indicates the reason for
     failure.  A call to CPXgeterrorstring will produce the text of
     the error message.  Note that CPXopenCPLEX produces no output,
     so the only way to see the cause of the error is to use
     CPXgeterrorstring.  For other CPLEX routines, the errors will
     be seen if the CPX_PARAM_SCRIND indicator is set to CPX_ON.  */

  if ( env == NULL ) {
    char  errmsg[1024];
    fprintf (stderr, "Could not open CPLEX environment.\n");
    CPXgeterrorstring (env, status, errmsg);
    fprintf (stderr, "%s", errmsg);
    exit(1);
  }

  /* Turn on output to the screen */

  status = CPXsetintparam (env, CPX_PARAM_SCRIND, CPX_ON);
  if ( status ) {
    fprintf (stderr,
             "Failure to turn on screen indicator, error %d.\n", status);
    exit(1);
  }

  //necessary when using own cut generator (or heuristic??) with CPLEX
  //status = CPXsetintparam (env, CPX_PARAM_PREIND, CPX_OFF);


  //set problem data

  lp = CPXcreateprob (env, &status, "ratio-lp");

  /* A returned pointer of NULL may mean that not enough memory
     was available or there was some other problem.  In the case of
     failure, an error message will have been written to the error
     channel from inside CPLEX.  In this example, the setting of
     the parameter CPX_PARAM_SCRIND causes the error message to
     appear on stdout.  */

  if ( lp == NULL ) {
    fprintf (stderr, "Failed to create LP.\n");
    exit(1);
  }

  /* Now copy the problem data into the lp */

  char* row_sense = new char[nConstraints];
  for (uint c=0; c < nConstraints; c++) {

    if (rhs_lower[c] == rhs_upper[c]) {
      row_sense[c] = 'E';
    }
    else {
      row_sense[c] = 'R';
    }
  }

  int* row_count = new int[nConstraints];
  for (uint c=0; c < nConstraints; c++)
    row_count[c] = row_start[c+1] - row_start[c];

  // status = CPXcopylp (env, lp, nVars, nConstraints, CPX_MIN, cost.direct_access(),
  // 		      rhs_upper.direct_access(), row_sense,
  // 		      (int*) row_start.direct_access(), row_count, (int*) lp_descr.col_indices(), lp_descr.value(),
  // 		      var_lb.direct_access(), var_ub.direct_access(), NULL);

  status = CPXnewcols (env, lp, nVars, cost.direct_access(), var_lb.direct_access(),
                       var_ub.direct_access(), NULL, NULL);
  if ( status )
    exit(1);

  std::cerr << "adding rows" << std::endl;

  CPXaddrows(env, lp, 0, nConstraints, lp_descr.nEntries(), rhs_lower.direct_access(), row_sense,
             (int*) row_start.direct_access(), (int*) lp_descr.col_indices(), lp_descr.value(),
             NULL, NULL);

  std::cerr << "setting ranges, nConstraints =  " << nConstraints << std::endl;

  for (int c=0; c < (int) nConstraints; c++) {
    if (row_sense[c] == 'R') {
      double range  = rhs_upper[c] - rhs_lower[c];

      //std::cerr << "row " << c << " has range " << range << std::endl;

      status = CPXchgrngval (env, lp, 1, &c, &range);
    }
  }

  delete[] row_sense;
  delete[] row_count;

  std::cerr << "calling optimize" << std::endl;

  //status = CPXmipopt (env, lp);
  status = CPXlpopt(env,lp);

  if ( status ) {
    fprintf (stderr, "Failed to optimize MIP.\n");
    exit(1);
  }

#else
  CoinPackedMatrix coinMatrix(false,(int*) lp_descr.row_indices(),(int*) lp_descr.col_indices(),
                              lp_descr.value(),lp_descr.nEntries());

  OsiClpSolverInterface lpSolver;
  lpSolver.loadProblem(coinMatrix, var_lb.direct_access(), var_ub.direct_access(),
                       cost.direct_access(), rhs_lower.direct_access(), rhs_upper.direct_access());
#endif
#endif

  /******* start the ratio minimization process ********/

  double cur_ratio = 0.0;

  Math1D::NamedVector<double> best_solution(nVars,0.0,MAKENAME(best_solution));

  uint iter = 0;

  while (true) {

    iter++;

    std::cerr << "+++++++++++++++ iteration #" << iter << " with ratio " << cur_ratio << std::endl;

    for (uint l=0; l < nLinePairs; l++)
      cost[l] = numerator_cost[l] - cur_ratio * denominator_cost[l];

#ifdef USE_GUROBI

    for (uint v=0; v < nVars; v++)
      GRBsetdblattrelement(grb_model,GRB_DBL_ATTR_OBJ,v,cost[v]);

    /* Optimize model */
    error = GRBoptimize(grb_model);
    assert(!error);

    double* lp_solution = new double[nVars];

    for (uint v=0; v < nVars; v++)
      GRBgetdblattrelement(grb_model,"X",v, lp_solution+v);

    double obj;
    GRBgetdblattr(grb_model, GRB_DBL_ATTR_OBJVAL, &obj);

#else

#ifdef USE_CPLEX
    Math1D::Vector<int> indices(nVars);
    for (uint v=0; v < nVars; v++)
      indices[v] = v;
    CPXchgobj(env, lp, nVars, indices.direct_access(), cost.direct_access());

    status = CPXlpopt(env,lp);

    double obj = 0.0;

    Math1D::Vector<double> cplex_solution(nVars);

    CPXsolution (env, lp, NULL, NULL, cplex_solution.direct_access(), NULL, NULL, NULL);

    const double* lp_solution = cplex_solution.direct_access();
#else
    lpSolver.setObjective(cost.direct_access());

    lpSolver.resolve();

    //     ClpSolve solve_options;
    //     solve_options.setSolveType(ClpSolve::useDual);
    //     solve_options.setPresolveType(ClpSolve::presolveNumber,5);
    //     lpSolver.setSolveOptions(solve_options);
    //     lpSolver.initialSolve();

    const double* lp_solution = lpSolver.getColSolution();

    double obj = lpSolver.getObjValue();
#endif
#endif

    if (obj >= -1e-6)
      break;

    double num = 0.0;
    double denom = 0.0;

    for (uint v=0; v < nVars; v++) {

      double val = lp_solution[v];
      best_solution[v] = val;

      num += val*numerator_cost[v];
      denom += val*denominator_cost[v];
    }

    std::cerr << "numerator: " << num << std::endl;
    std::cerr << "denominator: " << denom << std::endl;

    cur_ratio = num / denom;

    //TEMP
    Math1D::Vector<float> color(3);
    color[0] = 240.0;
    color[1] = 240.0;
    color[2] = 120.0;

    output.resize(xDim,yDim,3);
    if (zDim == 3)
      output = input;
    else {
      for (uint y=0; y < yDim; y++)
        for (uint x=0; x < xDim; x++)
          for (uint z=0; z < 3; z++)
            output(x,y,z) = input(x,y,0);
    }

    for (uint v=0; v < nVars; v++) {

      double val = best_solution[v];

      if (val >= 0.5) {
        const GridLinePair& pair = grid.get_line_pair(v);

        uint l1 = pair.line1_;
        uint l2 = pair.line2_;

        const GridLine& line1 = grid.get_line(l1);
        draw_line(output,line1.x1_,line1.y1_,line1.x2_,line1.y2_,color);

        const GridLine& line2 = grid.get_line(l2);
        draw_line(output,line2.x1_,line2.y1_,line2.x2_,line2.y2_,color);
      }
    }

    output.savePPM("im" + toString(iter) + ".ppm");
    //END_TEMP



#ifdef USE_GUROBI
    delete[] lp_solution;
#endif
  }

#ifdef USE_GUROBI
  GRBfreemodel(grb_model);
  GRBfreeenv(grb_env);
#endif
#ifdef USE_CPLEX
  CPXfreeprob (env, &lp);
  CPXcloseCPLEX (&env);
#endif

  output.resize(xDim,yDim,3);
  if (zDim == 3)
    output = input;
  else {
    for (uint y=0; y < yDim; y++)
      for (uint x=0; x < xDim; x++)
        for (uint z=0; z < 3; z++)
          output(x,y,z) = input(x,y,0);
  }

  uint nFrac = 0;

  Math1D::Vector<float> color(3);
  color[0] = 240.0;
  color[1] = 240.0;
  color[2] = 120.0;


#if 1
  for (uint v=0; v < nVars; v++) {

    double val = best_solution[v];

    if (val > 0.01 && val < 0.99)
      nFrac++;
    if (val >= 0.5) {
      const GridLinePair& pair = grid.get_line_pair(v);

      uint l1 = pair.line1_;
      uint l2 = pair.line2_;

      const GridLine& line1 = grid.get_line(l1);
      draw_line(output,line1.x1_,line1.y1_,line1.x2_,line1.y2_,color);

      //       uint x = std::min(xDim-1,line1.x1_);
      //       uint y = std::min(yDim-1,line1.y1_);

      //       output(x,y,0) = 240;
      //       output(x,y,1) = 240;
      //       output(x,y,2) = 120;

      //       x = std::min(xDim-1,line1.x2_);
      //       y = std::min(yDim-1,line1.y2_);

      //       output(x,y,0) = 240;
      //       output(x,y,1) = 240;
      //       output(x,y,2) = 120;

      const GridLine& line2 = grid.get_line(l2);
      draw_line(output,line2.x1_,line2.y1_,line2.x2_,line2.y2_,color);

      //       x = std::min(xDim-1,line2.x1_);
      //       y = std::min(yDim-1,line2.y1_);

      //       output(x,y,0) = 240;
      //       output(x,y,1) = 240;
      //       output(x,y,2) = 120;

      //       x = std::min(xDim-1,line2.x2_);
      //       y = std::min(yDim-1,line2.y2_);

      //       output(x,y,0) = 240;
      //       output(x,y,1) = 240;
      //       output(x,y,2) = 120;

    }
  }
#endif


  std::cerr << "solution contains " << nFrac << " fractional values" << std::endl;

}
