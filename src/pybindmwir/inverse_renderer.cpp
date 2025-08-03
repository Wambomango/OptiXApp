#include "pybindmwir/inverse_renderer.hpp"

namespace py = pybind11;


InverseRenderer::InverseRenderer()
{
    mwir_renderer_ = std::make_unique<MWIR::InverseRenderer>();
}

torch::Tensor InverseRenderer::Render(std::shared_ptr<Scene> scene, std::shared_ptr<ManyWorlds> many_worlds, std::optional<torch::Tensor> result_tensor)
{
    if (!mwir_renderer_)
    {
        throw std::runtime_error("InverseRenderer ownership has been transferred.");
    }

    return mwir_renderer_->Render(*scene->mwir_scene_, *many_worlds->mwir_many_worlds_, result_tensor);
}


void init_inverse_renderer(py::module_ &m)
{
    py::class_<InverseRenderer, std::shared_ptr<InverseRenderer>>(m, "InverseRenderer")
        .def(py::init<>())
        .def("Render", &InverseRenderer::Render);
}