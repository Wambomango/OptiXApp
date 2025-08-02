#pragma once

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mwir/forward_renderer.hpp"

#include "pybindmwir/scene.hpp"

#include <spdlog/spdlog.h>

namespace py = pybind11;

class ForwardRenderer
{

public:

    ForwardRenderer()
    {
        mwir_renderer_ = std::make_unique<MWIR::ForwardRenderer>();
    }
 
    torch::Tensor Render(std::shared_ptr<Scene> scene, std::optional<torch::Tensor> result_tensor = std::nullopt)
    {
        if (!mwir_renderer_)
        {
            throw std::runtime_error("ForwardRenderer ownership has been transferred.");
        }

        return mwir_renderer_->Render(*scene->mwir_scene_, result_tensor);
    }


private:
    std::unique_ptr<MWIR::ForwardRenderer> mwir_renderer_;
};

