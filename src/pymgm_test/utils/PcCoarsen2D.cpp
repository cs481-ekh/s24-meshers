/* PcCoarsen2D 
 * Coarsens a 2D point cloud 
 * xc = PcCoarsen2D(x);
 * 
 *
*/
#include <iostream>
#include <vector>
#include <random>
#include <array>
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
    std::vector <std::array<double, 2>> Coarsen(std::vector <cy::Point2d> points, int outputPointCount, float area) {

      // check arguments
      size_t N = points.size();
      if (static_cast<int>(N) <= outputPointCount) {
        throw std::invalid_argument("Output size must be less than input to coarsen");
      }

      uint idx = 0;

      // create weighted elimination object
      cy::WeightedSampleElimination<Point2d, double, 2, int> wse;

      // execute weighted elimination
      uint Nc = (uint) outputPointCount;
      std::vector < cy::Point2d > outputPoints(Nc);

      float d_max = 2 * wse.GetMaxPoissonDiskRadius(2, outputPoints.size(), area);

      bool isProgressive = true;
      wse.Eliminate(points.data(), points.size(),outputPoints.data(), outputPoints.size(), isProgressive, d_max, 2);

      Nc = outputPoints.size();

      std::vector <std::array<double, 2>> xc(Nc);

      for (idx = 0; idx < Nc; idx++) {
        xc[idx][0] = outputPoints[idx].x;
        xc[idx][1] = outputPoints[idx].y;
      }

      return xc;
    }


};