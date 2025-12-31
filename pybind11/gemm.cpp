#include <pybind11/pybind11.h>
#include "common.h"
#include "gemm.h"

namespace py = pybind11;

std::string gemm_backend_name(GemmBackend backend)
{
    switch (backend)
    {
    case GemmBackend::Naive:
        return "Naive (Hand-written matrix multiplication, no optimizations)";
    case GemmBackend::CuBLAS:
        return "CuBLAS (NVIDIA official GPU matrix multiplication library)";
    case GemmBackend::Opt1:
        return "Opt1 (Custom optimized matrix multiplication)";
    case GemmBackend::Opt2:
        return "Opt2 (Custom optimized matrix multiplication v2)";
    case GemmBackend::Opt3:
        return "Opt3 (Custom optimized matrix multiplication v3)";
    default:
        throw std::invalid_argument("Unknown GemmBackend enum value");
    }
}

PYBIND11_MODULE(gemm_ext, m)
{
    m.doc() = "CUDA GEMM benchmark module";

    // ================= Params =================
    py::class_<GemmParams>(m, "GemmParams")
        .def(py::init<>())
        .def(py::init<int, int, int>())  // 绑定带M、N、K参数的构造函数
        .def_readwrite("M", &GemmParams::M)
        .def_readwrite("N", &GemmParams::N)
        .def_readwrite("K", &GemmParams::K);

    // ================= Backend enum =================
    py::enum_<GemmBackend>(m, "GemmBackend")
        .value("Naive", GemmBackend::Naive)
        .value("CuBLAS", GemmBackend::CuBLAS)
        .value("Opt1", GemmBackend::Opt1)
        .value("Opt2", GemmBackend::Opt2)
        .value("Opt3", GemmBackend::Opt3)
        .export_values();

    m.def("gemm_backend_name", &gemm_backend_name, "获取 GemmBackend 枚举值的描述名称");

    // ================= Runner =================
    py::class_<GemmRunner>(m, "GemmRunner")
        .def(py::init<const GemmParams &, GemmBackend>(),
             py::arg("params"), py::arg("backend"))

        .def("init", &GemmRunner::init)
        .def("fill_random", &GemmRunner::fill_random,
             py::arg("seed") = 42)
        .def("run_once", &GemmRunner::run_once)
        .def("run_benchmark", &GemmRunner::run_benchmark,
             py::arg("iterations"))

        .def("fetch_result", &GemmRunner::fetch_result)
        .def("release", &GemmRunner::release)

        .def_static("compare",
                    &GemmRunner::compare,
                    py::arg("a"),
                    py::arg("b"),
                    py::arg("eps") = DEFAULT_EPS)

        .def_property_readonly("backend", &GemmRunner::backend)
        .def("print", &GemmRunner::print,
             py::arg("rows") = 4,
             py::arg("cols") = 4);
}
