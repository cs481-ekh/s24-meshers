#include <pybind11/pybind11.h>
#include "PcCoarsen2D.cpp"

namespace py = pybind11;
using namespace cy;

PYBIND11_MODULE(PcCoarsen, m) {
    py::class_<PcCoarsen2D>(m, "PcCoarsen2D")
        .def(py::init<>())
        .def("coarsen", &PcCoarsen2D::Coarsen);
}


