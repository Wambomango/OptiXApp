#pragma once

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mwir/renderer.hpp"

#include "pybindmwir/scene.hpp"


class Renderer
{
public:

    Renderer();

    torch::Tensor Render(std::shared_ptr<Scene> scene, std::optional<torch::Tensor> result_tensor = std::nullopt, std::optional<int> seed = std::nullopt);

private:
    std::unique_ptr<MWIR::Renderer> mwir_renderer_;
};

void init_renderer(pybind11::module_ &);
