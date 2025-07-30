#pragma once

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mwir/include/renderer.hpp"

#include "pybindmwir/scene.hpp"

#include <spdlog/spdlog.h>

namespace py = pybind11;

class Renderer
{

public:

    Renderer(std::shared_ptr<Scene> scene)
    {
        mwir_renderer_ = std::make_unique<MWIR::Renderer>(std::move(*scene->mwir_scene_));
        scene->mwir_scene_.reset();
    }
 
    ~Renderer()
    {
    }

    void SetScene(std::shared_ptr<Scene> scene)
    {
        if (!mwir_renderer_)
        {
            throw std::runtime_error("Renderer ownership has been transferred.");
        }

        mwir_renderer_->SetScene(std::move(*(scene->mwir_scene_)));
        scene->mwir_scene_.reset();
    }

    std::shared_ptr<Scene> GetScene()
    {
        if (!mwir_renderer_)
        {
            throw std::runtime_error("Renderer ownership has been transferred.");
        }
        std::shared_ptr<Scene> scene = std::make_shared<Scene>(mwir_renderer_->GetScene());
        return scene;
    }

    at::Tensor Render()
    {
        if (!mwir_renderer_)
        {
            throw std::runtime_error("Renderer ownership has been transferred.");
        }

        return mwir_renderer_->Render();
    }


protected:
    std::unique_ptr<MWIR::Renderer> mwir_renderer_;
};

