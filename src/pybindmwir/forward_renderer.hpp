#pragma once

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mwir/forward_renderer.hpp"

#include "pybindmwir/scene.hpp"


class ForwardRenderer
{
public:

    ForwardRenderer();

    torch::Tensor Render(std::shared_ptr<Scene> scene, std::optional<torch::Tensor> result_tensor = std::nullopt);

private:
    std::unique_ptr<MWIR::ForwardRenderer> mwir_renderer_;
};

void init_forward_renderer(pybind11::module_ &);
