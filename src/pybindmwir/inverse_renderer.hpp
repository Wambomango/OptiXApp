#pragma once

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mwir/inverse_renderer.hpp"

#include "pybindmwir/scene.hpp"
#include "pybindmwir/many_worlds.hpp"

#include <spdlog/spdlog.h>

class InverseRenderer
{

public:

    InverseRenderer();

    torch::Tensor Render(std::shared_ptr<Scene> scene, std::shared_ptr<ManyWorlds> many_worlds, std::optional<torch::Tensor> result_tensor = std::nullopt);


private:
    std::unique_ptr<MWIR::InverseRenderer> mwir_renderer_;
};


void init_inverse_renderer(pybind11::module_ &m);