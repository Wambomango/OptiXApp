#pragma once

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mwir/include/forward_renderer.hpp"

#include "pybindmwir/scene.hpp"

#include <spdlog/spdlog.h>

namespace py = pybind11;

class ForwardRenderer
{

public:

    ForwardRenderer(std::shared_ptr<Scene> scene)
    {
        mwir_renderer_ = std::make_unique<MWIR::ForwardRenderer>(std::move(*scene->mwir_scene_));
        scene->mwir_scene_.reset();
    }
 
    ~ForwardRenderer()
    {
    }

    void SetScene(std::shared_ptr<Scene> scene)
    {
        if (!mwir_renderer_)
        {
            throw std::runtime_error("ForwardRenderer ownership has been transferred.");
        }

        mwir_renderer_->SetScene(std::move(*(scene->mwir_scene_)));
        scene->mwir_scene_.reset();
    }

    std::shared_ptr<Scene> GetScene()
    {
        if (!mwir_renderer_)
        {
            throw std::runtime_error("ForwardRenderer ownership has been transferred.");
        }
        std::shared_ptr<Scene> scene = std::make_shared<Scene>(mwir_renderer_->GetScene());
        return scene;
    }

    at::Tensor Render()
    {
        if (!mwir_renderer_)
        {
            throw std::runtime_error("ForwardRenderer ownership has been transferred.");
        }

        return mwir_renderer_->Render();
    }


protected:
    std::unique_ptr<MWIR::ForwardRenderer> mwir_renderer_;
};

