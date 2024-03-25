#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // Include the stl header for vector conversion
#include "PcCoarsen2D.cpp"

namespace py = pybind11;
using namespace cy;

PYBIND11_MODULE(PcCoarsen, m) {
    py::class_<PcCoarsen2D>(m, "PcCoarsen2D")
        .def(py::init<>())
        .def("Coarsen", &PcCoarsen2D::Coarsen);

    py::class_<WeightedSampleElimination<cy::Point2d, double, 2, int>>(m, "WeightedSampleElimination")
        .def(py::init<>())
        .def("SetTiling", &WeightedSampleElimination<cy::Point2d, double, 2, int>::SetTiling, py::arg("on")=true)  // Bind the SetTiling function
        .def("IsTiling", &WeightedSampleElimination<cy::Point2d, double, 2, int>::IsTiling)
        .def("SetWeightLimiting", &WeightedSampleElimination<cy::Point2d, double, 2, int>::SetWeightLimiting, py::arg("on")=true)
        .def("IsWeightLimiting", &WeightedSampleElimination<cy::Point2d, double, 2, int>::IsWeightLimiting)
        .def("GetParamAlpha", &WeightedSampleElimination<cy::Point2d, double, 2, int>::GetParamAlpha)
        .def("GetMaxPoissonDiskRadius", &WeightedSampleElimination<cy::Point2d, double, 2, int>::GetMaxPoissonDiskRadius,
              py::arg("dimensions"), py::arg("sampleCount"), py::arg("domainSize")=0.0f);
}


