#pragma once

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mwir/many_worlds_renderer.hpp"

#include "pybindmwir/scene.hpp"
#include "pybindmwir/many_worlds.hpp"

#include <spdlog/spdlog.h>

class ManyWorldsRenderer
{

public:

    ManyWorldsRenderer();

    torch::Tensor Forward(std::shared_ptr<Scene> scene, std::shared_ptr<ManyWorlds> many_worlds, std::optional<torch::Tensor> result_tensor = std::nullopt, std::optional<int> seed = std::nullopt);
    void Backward(std::shared_ptr<Scene> scene, std::shared_ptr<ManyWorlds> many_worlds, torch::Tensor grad_output, std::optional<int> seed = std::nullopt);


private:
    std::unique_ptr<MWIR::ManyWorldsRenderer> mwir_renderer_;
};


void init_many_worlds_renderer(pybind11::module_ &m);