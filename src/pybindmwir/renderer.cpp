#include "pybindmwir/renderer.hpp"

namespace py = pybind11;


Renderer::Renderer()
{
    mwir_renderer_ = std::make_unique<MWIR::Renderer>();
}
 
torch::Tensor Renderer::Render(std::shared_ptr<Scene> scene, std::optional<torch::Tensor> result_tensor, std::optional<int> seed)
{
    if (!mwir_renderer_)
    {
        throw std::runtime_error("Renderer ownership has been transferred.");
    }

    return mwir_renderer_->Render(*scene->mwir_scene_, result_tensor, seed);
}


void init_renderer(py::module_ &m)
{
    
    py::class_<Renderer, std::shared_ptr<Renderer>>(m, "Renderer")
        .def(py::init<>())
        .def("Render", &Renderer::Render);

}
