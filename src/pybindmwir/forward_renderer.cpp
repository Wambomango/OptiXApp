#include "pybindmwir/forward_renderer.hpp"

namespace py = pybind11;


ForwardRenderer::ForwardRenderer()
{
    mwir_renderer_ = std::make_unique<MWIR::ForwardRenderer>();
}
 
torch::Tensor ForwardRenderer::Render(std::shared_ptr<Scene> scene, std::optional<torch::Tensor> result_tensor)
{
    if (!mwir_renderer_)
    {
        throw std::runtime_error("ForwardRenderer ownership has been transferred.");
    }

    return mwir_renderer_->Render(*scene->mwir_scene_, result_tensor);
}


void init_forward_renderer(py::module_ &m)
{
    
    py::class_<ForwardRenderer, std::shared_ptr<ForwardRenderer>>(m, "ForwardRenderer")
        .def(py::init<>())
        .def("Render", &ForwardRenderer::Render);

}
