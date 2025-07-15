#include "mwir/include/renderer.hpp"
#include "mwir/renderer_impl.hpp"

namespace MWIR
{

Renderer::Renderer(Scene &&scene)
{
    impl = new RendererImpl(std::move(*scene.impl));
    scene.impl = nullptr;
}

Renderer::~Renderer()
{
    if (impl)
    {
        delete impl;
    }
}

Renderer::Renderer(Renderer&& other) noexcept : impl(std::move(other.impl))
{
    other.impl = nullptr; 
}

Renderer& Renderer::operator=(Renderer&& other) noexcept
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

void Renderer::SetScene(Scene&& scene)
{
    if (impl)
    {
        impl->SetScene(std::move(*scene.impl));
        scene.impl = nullptr;
    }
}

at::Tensor Renderer::Render()
{
    if (impl)
    {
        return impl->Render();
    }
    return at::Tensor();
} 




}