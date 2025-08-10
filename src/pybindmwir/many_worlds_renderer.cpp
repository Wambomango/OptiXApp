#include "pybindmwir/many_worlds_renderer.hpp"

namespace py = pybind11;


ManyWorldsRenderer::ManyWorldsRenderer()
{
    mwir_renderer_ = std::make_unique<MWIR::ManyWorldsRenderer>();
}

torch::Tensor ManyWorldsRenderer::Forward(std::shared_ptr<Scene> scene, std::shared_ptr<ManyWorlds> many_worlds, std::optional<torch::Tensor> result_tensor, std::optional<int> seed)
{
    if (!mwir_renderer_)
    {
        throw std::runtime_error("ManyWorldsRenderer ownership has been transferred.");
    }

    return mwir_renderer_->Forward(*scene->mwir_scene_, *many_worlds->mwir_many_worlds_, result_tensor, seed);
}

void ManyWorldsRenderer::Backward(std::shared_ptr<Scene> scene, std::shared_ptr<ManyWorlds> many_worlds, torch::Tensor grad_output, std::optional<int> seed)
{
    if (!mwir_renderer_)
    {
        throw std::runtime_error("ManyWorldsRenderer ownership has been transferred.");
    }

    return mwir_renderer_->Backward(*scene->mwir_scene_, *many_worlds->mwir_many_worlds_, grad_output, seed);
}


void init_many_worlds_renderer(py::module_ &m)
{
    py::class_<ManyWorldsRenderer, std::shared_ptr<ManyWorldsRenderer>>(m, "ManyWorldsRenderer")
        .def(py::init<>())
        .def("Forward", &ManyWorldsRenderer::Forward)
        .def("Backward", &ManyWorldsRenderer::Backward);
}