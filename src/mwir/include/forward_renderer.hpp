#pragma once

#include "scene.hpp"

#include <torch/torch.h>

namespace MWIR
{

class ForwardRendererImpl;

class ForwardRenderer
{

public:
    ForwardRenderer(Scene &&scene);
    ~ForwardRenderer();    
    ForwardRenderer(const ForwardRenderer&) = delete;
    ForwardRenderer& operator=(const ForwardRenderer&) = delete;
    ForwardRenderer(ForwardRenderer&&) noexcept;
    ForwardRenderer& operator=(ForwardRenderer&&) noexcept;

    void SetScene(Scene&& scene);
    Scene GetScene();
    at::Tensor Render();


private:
    ForwardRendererImpl *impl;
};

}