#include "mwir/include/forward_renderer.hpp"
#include "mwir/forward_renderer_impl.hpp"

namespace MWIR
{

ForwardRenderer::ForwardRenderer(Scene &&scene)
{
    impl = new ForwardRendererImpl(std::move(*scene.impl));
    scene.impl = nullptr;
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

void ForwardRenderer::SetScene(Scene&& scene)
{
    if(!impl)
    {
        throw std::runtime_error("ForwardRenderer ownership has been transferred");
    }

    impl->SetScene(std::move(*scene.impl));
    scene.impl = nullptr;
}

Scene ForwardRenderer::GetScene()
{
    if (!impl)
    {
        throw std::runtime_error("ForwardRenderer ownership is not initialized.");
    }

    Scene tmp(impl->GetScene());
    return tmp;
}

at::Tensor ForwardRenderer::Render()
{
    if (!impl)
    {
        throw std::runtime_error("ForwardRenderer ownership is not initialized.");
    }
    
    return impl->Render();
}




}