/* PcCoarsen2D 
 * Coarsens a 2D point cloud 
 * xc = PcCoarsen2D(x);
 * 
 * To create the mex file for the function
 * 1. Open matlab and change directories to where this file is saved
 * 2. Type "mex -I./cyCodeBase/ PcCoarsen2D.cpp" on the command line
 *    Change "-I./cyCodeBase/" to point to where you have the cyCodeBase package installed
 *
 * When compiling on AppleSilicon (M1 or M2), I found it necessary to include
 * the -DCY_NO_EMMINTRIN_H option with the mex compilation.
*/
#include <iostream>
#include <vector>
#include <random>
#include "cySampleElim.h"
#include "cyPointCloud.h"
#include "cyPoint.h"
#include "cyHeap.h"
#include "mex.hpp"
#include "mexAdapter.hpp"
#include "MatlabDataArray.hpp"

using namespace matlab::data;
using matlab::mex::ArgumentList;
using namespace cy;
typedef unsigned int uint;

// # define M_PI	3.14159265358979323846

class MexFunction : public matlab::mex::Function {
public:
    void operator()(ArgumentList outputs, ArgumentList inputs) {
        // checkArguments(outputs, inputs);
        uint Nc = (uint) inputs[1][0];

        TypedArray<double> doubleArray = std::move(inputs[0]);
        int N = doubleArray.getNumberOfElements()/2;
        std::vector< cy::Point2d > inputPoints(N);
		// int count = 0;
		int idx = 0; 
		for ( idx = 0; idx < N; idx++ ) {
		    inputPoints[idx].x = doubleArray[idx][0];
		    inputPoints[idx].y = doubleArray[idx][1];
		}        

		// create weighted elimination object
		cy::WeightedSampleElimination< Point2d, double, 2, int > wse;

		// execute weighted elimination
		std::vector< cy::Point2d > outputPoints(Nc);
        float area = inputs[2][0];
		float d_max = 2 * wse.GetMaxPoissonDiskRadius( 
									2, outputPoints.size(), area );
		
		bool isProgressive = true;
		wse.Eliminate( inputPoints.data(), inputPoints.size(), 
		               outputPoints.data(), outputPoints.size(), 
		               isProgressive, d_max, 2 );

		Nc = outputPoints.size();

		ArrayFactory f;
		TypedArray<double> xc = f.createArray<double>({Nc, 2});
		for ( idx = 0; idx < Nc; idx++ ) {
		    xc[idx][0] = outputPoints[idx].x;
		    xc[idx][1] = outputPoints[idx].y;
		}        
		outputs[0] = xc;
    }

    void checkArguments(ArgumentList outputs, ArgumentList inputs) {
        // Get pointer to engine
        std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();

        // Get array factory
        ArrayFactory factory;

        // Check first input argument
        if (inputs[0].getType() != ArrayType::DOUBLE ||
            inputs[0].getType() == ArrayType::COMPLEX_DOUBLE ||
            inputs[0].getNumberOfElements() != 1)
        {
            matlabPtr->feval(u"error",
                0,
                std::vector<Array>({ factory.createScalar("First input must be scalar double") }));
        }

        // Check second input argument
        if (inputs[1].getType() != ArrayType::DOUBLE ||
            inputs[1].getType() == ArrayType::COMPLEX_DOUBLE)
        {
            matlabPtr->feval(u"error",
                0,
                std::vector<Array>({ factory.createScalar("Input must be double array") }));
        }
        // Check number of outputs
        if (outputs.size() > 1) {
            matlabPtr->feval(u"error",
                0,
                std::vector<Array>({ factory.createScalar("Only one output is returned") }));
        }
    }
};
