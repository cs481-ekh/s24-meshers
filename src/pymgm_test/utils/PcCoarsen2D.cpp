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

// inputPoints is input vector array of (x,y) inputPoints.
// Want to coarsen (eliminate) down to size outputPointCount
// Calls wse.eliminate() which loops and eliminates one point at a time until outputPointCount is reached
// return that vector array
class PcCoarsen2D {
public:

    std::vector <std::array<double, 2>> Coarsen(std::vector <std::array<double, 2>> inputPoints, int outputPointCount, float area) {

      //Cast inputPoints from type array <double, 2> to type vector< Point2d >
      TypedArray<double> doubleArray = std::move(inputs[0]);
      int N = inputPoints.size();
      std::vector< cy::Point2d > cyInputPoints(N);
      for ( int i = 0; i < N; i++ ) {
        cyInputPoints[i].x = inputPoints[i][0];
        cyInputPoints[i].y = inputPoints[i][1];
      }

      // create weighted elimination object
      cy::WeightedSampleElimination<Point2d, double, 2, int> wse;

      // execute weighted elimination
      std::vector < cy::Point2d > outputPoints(outputPointCount);

      float d_max = 2 * wse.GetMaxPoissonDiskRadius(2, outputPoints.size(), area);

      bool isProgressive = true;
      wse.Eliminate(cyInputPoints.data(), cyInputPoints.size(),outputPoints.data(), outputPoints.size(), isProgressive, d_max, 2);
      
      // will the data types break here? unsure if outputPoints from eliminate() is of type int or (uint) ? 
      outputPointCount = outputPoints.size();  // reassign just to be sure? Could eliminite() ever be off by one?

      std::vector <std::array<double, 2>> xc(outputPointCount);

      for (int i = 0; i < outputPointCount; i++) {
        xc[i][0] = outputPoints[i].x;
        xc[i][1] = outputPoints[i].y;
      }

      return xc;
    }


};