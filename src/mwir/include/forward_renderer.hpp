#pragma once

#include "scene.hpp"

#include <torch/torch.h>

namespace MWIR
{

class ForwardRendererImpl;

class ForwardRenderer
{

public:
    ForwardRenderer();
    ~ForwardRenderer();    
    ForwardRenderer(const ForwardRenderer&) = delete;
    ForwardRenderer& operator=(const ForwardRenderer&) = delete;
    ForwardRenderer(ForwardRenderer&&) noexcept;
    ForwardRenderer& operator=(ForwardRenderer&&) noexcept;

    at::Tensor Render(Scene &scene, std::optional<at::Tensor> result_tensor = std::nullopt);


private:
    ForwardRendererImpl *impl;
};

}