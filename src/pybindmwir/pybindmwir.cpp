
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pybindmwir/antenna.hpp"
#include "pybindmwir/mesh.hpp"
#include "pybindmwir/forward_renderer.hpp"
#include "pybindmwir/scene.hpp"
#include "pybindmwir/signal.hpp"


PYBIND11_MODULE(PyBindMWIR, m) 
{
    m.doc() = "PyBindMWIR python module";

    py::class_<Antenna, std::shared_ptr<Antenna>>(m, "Antenna")
        .def(py::init<>())
        .def(py::init<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&>())
        .def("SetPosition", &Antenna::SetPosition)
        .def("SetOrientation", &Antenna::SetOrientation)
        .def("SetFOV", &Antenna::SetFOV)
        .def("SetRayDensity", &Antenna::SetRayDensity)
        .def("GetPosition", &Antenna::GetPosition)
        .def("GetOrientation", &Antenna::GetOrientation)
        .def("GetFOV", &Antenna::GetFOV)
        .def("GetRayDensity", &Antenna::GetRayDensity)
        .def("GetSolidAngle", &Antenna::GetSolidAngle)
        .def("GetNRays", &Antenna::GetNRays);

    py::class_<Mesh, std::shared_ptr<Mesh>>(m, "Mesh")
        .def(py::init<>())
        .def(py::init<torch::Tensor&>())
        .def("SetVertices", &Mesh::SetVertices)
        .def("GetVertices", &Mesh::GetVertices);

    py::class_<ForwardRenderer, std::shared_ptr<ForwardRenderer>>(m, "ForwardRenderer")
        .def(py::init<std::shared_ptr<Scene>>())
        .def("SetScene", &ForwardRenderer::SetScene)
        .def("GetScene", &ForwardRenderer::GetScene)
        .def("Render", &ForwardRenderer::Render);

    py::class_<Scene, std::shared_ptr<Scene>>(m, "Scene")
        .def(py::init<>())
        .def(py::init<std::shared_ptr<Mesh>, std::vector<std::shared_ptr<Antenna>>, std::vector<std::shared_ptr<Antenna>>, std::shared_ptr<Signal>>())
        .def("SetMesh", &Scene::SetMesh)
        .def("SetSenders", &Scene::SetSenders)
        .def("SetReceivers", &Scene::SetReceivers)
        .def("SetSignal", &Scene::SetSignal)
        .def("GetMesh", &Scene::GetMesh)
        .def("GetSenders", &Scene::GetSenders)
        .def("GetReceivers", &Scene::GetReceivers)
        .def("GetSignal", &Scene::GetSignal);

    py::class_<Signal, std::shared_ptr<Signal>>(m, "Signal")
        .def(py::init<>())
        .def(py::init<torch::Tensor&, torch::Tensor&>())
        .def("SetFrequencyRange", &Signal::SetFrequencyRange)
        .def("GetFrequencyRange", &Signal::GetFrequencyRange)
        .def("GetNSamples", &Signal::GetNSamples)
        .def("GetFStep", &Signal::GetFStep);

}





