#include "mwir/include/forward_renderer.hpp"
#include "mwir/forward_renderer_impl.hpp"

namespace MWIR
{

ForwardRenderer::ForwardRenderer()
{
    impl = new ForwardRendererImpl();
}

ForwardRenderer::~ForwardRenderer()
{
    if (impl)
    {
        delete impl;
    }
}

ForwardRenderer::ForwardRenderer(ForwardRenderer&& other) noexcept : impl(std::move(other.impl))
{
    other.impl = nullptr; 
}

ForwardRenderer& ForwardRenderer::operator=(ForwardRenderer&& other) noexcept
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

at::Tensor ForwardRenderer::Render(Scene &scene, std::optional<at::Tensor> result_tensor)
{
    if (!impl)
    {
        throw std::runtime_error("ForwardRenderer ownership is not initialized.");
    }

    return impl->Render(*scene.impl, result_tensor);
}

}
