#pragma once

#include "scene.hpp"

#include <memory>

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
    Renderer(Renderer&&) = default;
    Renderer& operator=(Renderer&&) = default;

    void SetScene(Scene&& scene);
    void Render(int n_rays, int batch_size);

private:
    std::unique_ptr<RendererImpl> impl;
};

}