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
    if(!impl)
    {
        throw std::runtime_error("Renderer ownership has been transferred");
    }

    impl->SetScene(std::move(*scene.impl));
    scene.impl = nullptr;
}

Scene Renderer::GetScene()
{
    if (!impl)
    {
        throw std::runtime_error("Renderer ownership is not initialized.");
    }

    Scene tmp(impl->GetScene());
    return tmp;
}

at::Tensor Renderer::Render()
{
    if (!impl)
    {
        throw std::runtime_error("Renderer ownership is not initialized.");
    }
    
    return impl->Render();
}




}