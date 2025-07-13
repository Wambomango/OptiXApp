#include "mwir/include/renderer.hpp"
#include "mwir/renderer_impl.hpp"

namespace MWIR
{

Renderer::Renderer(Scene &&scene)
{
    impl = std::make_unique<RendererImpl>(std::move(*scene.impl));
    scene.impl.reset();
}

Renderer::~Renderer()
{
}

void Renderer::SetScene(Scene&& scene)
{
    impl->SetScene(std::move(*scene.impl));
    scene.impl.reset();
}

void Renderer::Render(int n_rays, int batch_size)
{
    impl->Render(n_rays, batch_size);
}

}