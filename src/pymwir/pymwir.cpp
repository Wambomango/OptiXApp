
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pymwir/antenna.hpp"
// #include "pymwir/mesh.hpp"
// #include "pymwir/scene.hpp"
// #include "pymwir/signal.hpp"


PYBIND11_MODULE(PyMWIR, m) 
{
    m.doc() = "PyMWIR python module";

    py::class_<Antenna>(m, "Antenna")
        .def(py::init<at::Tensor&, at::Tensor&, at::Tensor&>())
        .def("SetPosition", &Antenna::SetPosition)
        .def("SetOrientation", &Antenna::SetOrientation)
        .def("SetFOV", &Antenna::SetFOV)
        .def("GetPosition", &Antenna::GetPosition)
        .def("GetOrientation", &Antenna::GetOrientation)
        .def("GetFOV", &Antenna::GetFOV);

//     py::class_<Mesh>(m, "Mesh")
//         .def(py::init<torch::Tensor&>())
//         .def("SetVertices", &Mesh::SetVertices);

//     py::class_<Scene>(m, "Scene")
//         .def(py::init<std::shared_ptr<Mesh>, std::vector<std::shared_ptr<Antenna>>, std::vector<std::shared_ptr<Antenna>>, std::shared_ptr<Signal>>());
//         // .def("SetMesh", &Scene::SetMesh)
//         // .def("SetSenders", &Scene::SetSenders)
//         // .def("SetReceivers", &Scene::SetReceivers)
//         // .def("SetSignal", &Scene::SetSignal);

//     py::class_<Signal>(m, "Signal")
//         .def(py::init<torch::Tensor&, torch::Tensor&>())
//         .def("GetFrequencyRange", &Signal::GetFrequencyRange)
//         .def("GetNFrequencies", &Signal::GetNFrequencies)
//         .def("GetFStep", &Signal::GetFStep);

}





