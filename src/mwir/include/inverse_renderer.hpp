#pragma once

#include "scene.hpp"

#include <torch/torch.h>

namespace MWIR
{

class InverseRendererImpl;

class InverseRenderer
{

public:
    InverseRenderer();
    ~InverseRenderer();
    InverseRenderer(const InverseRenderer&) = delete;
    InverseRenderer& operator=(const InverseRenderer&) = delete;
    InverseRenderer(InverseRenderer&&) noexcept;
    InverseRenderer& operator=(InverseRenderer&&) noexcept;

    at::Tensor Render(Scene &scene, std::optional<at::Tensor> result_tensor = std::nullopt);


private:
    InverseRendererImpl *impl;
};

}