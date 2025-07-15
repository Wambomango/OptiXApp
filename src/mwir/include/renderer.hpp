#pragma once

#include "scene.hpp"

#include <torch/torch.h>

namespace MWIR
{

class RendererImpl;
  
class Renderer
{

public:
    Renderer(Scene &&scene);
    ~Renderer();    
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    Renderer(Renderer&&) noexcept;
    Renderer& operator=(Renderer&&) noexcept;

    void SetScene(Scene&& scene);
    at::Tensor Render();

private:
    RendererImpl *impl;
};

}