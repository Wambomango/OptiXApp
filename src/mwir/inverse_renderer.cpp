#include "mwir/include/inverse_renderer.hpp"
#include "mwir/inverse_renderer_impl.hpp"

namespace MWIR
{

InverseRenderer::InverseRenderer()
{
    impl = new InverseRendererImpl();
}

InverseRenderer::~InverseRenderer()
{
    if (impl)
    {
        delete impl;
    }
}

InverseRenderer::InverseRenderer(InverseRenderer&& other) noexcept : impl(std::move(other.impl))
{
    other.impl = nullptr; 
}

InverseRenderer& InverseRenderer::operator=(InverseRenderer&& other) noexcept
{
    if (this != &other)
    {
        if(impl)
        {
            delete impl; 
        }
        impl = other.impl; 
        other.impl = nullptr; 
    }
    return *this;
}


at::Tensor InverseRenderer::Render(Scene &scene, std::optional<at::Tensor> result_tensor)
{
    if (!impl)
    {
        throw std::runtime_error("InverseRenderer ownership is not initialized.");
    }

    return impl->Render(*scene.impl, result_tensor);
}


}