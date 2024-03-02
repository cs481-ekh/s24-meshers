#include <pybind11/pybind11.h>
#include "cyPointCloud.h"

namespace py = pybind11;
using namespace cy;

PYBIND11_MODULE(cyPCModule, m) {
    py::class_<PointCloud>(m, "PointCloud")
        .def(py::init<>())
        .def("getPointCount", &PointCloud::GetPointCount);
}
