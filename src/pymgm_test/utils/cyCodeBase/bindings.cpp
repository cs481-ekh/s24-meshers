#include <pybind11/pybind11.h>
#include "cySampleElim.h"
#include "cyPoint.h"

namespace py = pybind11;
using namespace cy;

PYBIND11_MODULE(elim, m) {
    py::class_<WeightedSampleElimination<cy::Point2d, double, 2, int>>(m, "WeightedSampleElimination")
		.def(py::init<>())  // Default constructor
		.def("IsTiling", &WeightedSampleElimination<cy::Point2d, double, 2, int>::IsTiling);

    // You can add bindings for other member functions here if needed
}
