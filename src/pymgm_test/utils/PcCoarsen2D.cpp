/* PcCoarsen2D 
 * Coarsens a 2D point cloud 
 * xc = PcCoarsen2D(x);
 * 
 *
*/
#include <iostream>
#include <vector>
#include <random>
#include "cyCodeBase/cySampleElim.h"
#include "cyCodeBase/cyPointCloud.h"
#include "cyCodeBase/cyPoint.h"
#include "cyCodeBase/cyHeap.h"

using namespace cy;
typedef unsigned int uint;

class PcCoarsen2D {
public:
    // points is input array.
    // Want to coarsen to a vector array of (x,y) points coarsened to size outputPointCount
    // Calls wse.eliminate() which loops and eliminates one point at a time until outputPointCount is reached
    // return the vector array
    std::vector <std::array<double, 2>> Coarsen(vector <cy::Point2d> points, int outputPointCount, float area) {
      // check arguments
      if (points.getType() != ArrayType::DOUBLE || points.getType() == ArrayType::COMPLEX_DOUBLE)
      {
        throw std::invalid_argument("Input points must be scalar double ");
      }

      if (outputPointCount.getType() != ArrayType::DOUBLE)
      {
        throw std::invalid_argument("outputPointCount must be int ????? ");
      }

      TypedArray<double> doubleArray = std::move(points);
      int N = doubleArray.getNumberOfElements()/2;
      if (N <= outputPointCount) {
        throw std::invalid_argument("Output size must be less than input to coarsen");
      }

      std::vector <cy::Point2d> inputPoints(N);
      int idx = 0;
      for (idx = 0; idx < N; idx++) {
        inputPoints[idx].x = doubleArray[idx][0];
        inputPoints[idx].y = doubleArray[idx][1];
      }

      // create weighted elimination object
      cy::WeightedSampleElimination<Point2d, double, 2, int> wse;

      // execute weighted elimination
      uint Nc = (uint) outputPointCount;
      std::vector < cy::Point2d > outputPoints(Nc);

      float d_max = 2 * wse.GetMaxPoissonDiskRadius(2, outputPoints.size(), area);

      bool isProgressive = true;
      wse.Eliminate(inputPoints.data(), inputPoints.size(),outputPoints.data(), outputPoints.size(), isProgressive, d_max, 2);

      Nc = outputPoints.size();

      std::vector <std::array<double, 2>> xc(Nc);

      for (idx = 0; idx < Nc; idx++) {
        xc[idx][0] = outputPoints[idx].x;
        xc[idx][1] = outputPoints[idx].y;
      }

      return xc;
    }


};