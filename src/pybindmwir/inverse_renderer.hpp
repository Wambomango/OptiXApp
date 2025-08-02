#pragma once

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mwir/inverse_renderer.hpp"

#include "pybindmwir/scene.hpp"
#include "pybindmwir/many_worlds.hpp"

#include <spdlog/spdlog.h>

namespace py = pybind11;

class InverseRenderer
{

public:

    InverseRenderer()
    {
        mwir_renderer_ = std::make_unique<MWIR::InverseRenderer>();
    }

    torch::Tensor Render(std::shared_ptr<Scene> scene, std::shared_ptr<ManyWorlds> many_worlds, std::optional<torch::Tensor> result_tensor = std::nullopt)
    {
        if (!mwir_renderer_)
        {
            throw std::runtime_error("InverseRenderer ownership has been transferred.");
        }

        return mwir_renderer_->Render(*scene->mwir_scene_, *many_worlds->mwir_many_worlds_, result_tensor);
    }


private:
    std::unique_ptr<MWIR::InverseRenderer> mwir_renderer_;
};

