#pragma once

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mwir/include/antenna.hpp"

namespace py = pybind11;

class Antenna
{

public:
    Antenna()
    {
        mwir_antenna_ = std::make_unique<MWIR::Antenna>();
    }

    Antenna(MWIR::Antenna &&mwir_antenna)
    {
        mwir_antenna_ = std::make_unique<MWIR::Antenna>(std::move(mwir_antenna));
    }

    Antenna(at::Tensor &position, at::Tensor &euler, at::Tensor &fov, at::Tensor &ray_density)
    {
        mwir_antenna_ = std::make_unique<MWIR::Antenna>(
            glm::vec3(position[0].item<float>(), position[1].item<float>(), position[2].item<float>()),
            glm::vec3(euler[0].item<float>(), euler[1].item<float>(), euler[2].item<float>()),
            glm::vec2(fov[0].item<float>(), fov[1].item<float>()), ray_density[0].item<float>());
    }
 
    ~Antenna()
    {
    }

    void SetPosition(const at::Tensor &position)
    {
        if (!mwir_antenna_)
        {
            throw std::runtime_error("Antenna ownership has been transferred.");
        }

        glm::vec3 pos = {position[0].item<float>(), position[1].item<float>(), position[2].item<float>()};
        mwir_antenna_->SetPosition(pos);
    } 

    void SetOrientation(const at::Tensor &euler)
    {
        if (!mwir_antenna_)
        {
            throw std::runtime_error("Antenna ownership has been transferred.");
        }

        glm::vec3 euler_vec = {euler[0].item<float>(), euler[1].item<float>(), euler[2].item<float>()};
        mwir_antenna_->SetOrientation(euler_vec);
    }

    void SetFOV(const at::Tensor &fov)
    {
        if (!mwir_antenna_)
        {
            throw std::runtime_error("Antenna ownership has been transferred.");
        }

        glm::vec2 fov_vec = {fov[0].item<float>(), fov[1].item<float>()};
        mwir_antenna_->SetFOV(fov_vec);
    }

    void SetRayDensity(const at::Tensor &ray_density)
    {
        if (!mwir_antenna_)
        {
            throw std::runtime_error("Antenna ownership has been transferred.");
        }

        mwir_antenna_->SetRayDensity(ray_density[0].item<float>());
    }

    at::Tensor GetPosition() const
    {
        if (!mwir_antenna_)
        {
            throw std::runtime_error("Antenna ownership has been transferred.");
        }

        glm::vec3 pos = mwir_antenna_->GetPosition();
        return at::tensor({pos.x, pos.y, pos.z}, torch::kFloat32).view({3});
    }

    at::Tensor GetOrientation() const
    {
        if (!mwir_antenna_)
        {
            throw std::runtime_error("Antenna ownership has been transferred.");
        }

        glm::vec3 euler = mwir_antenna_->GetOrientation();
        return at::tensor({euler.x, euler.y, euler.z}, torch::kFloat32).view({3});
    }

    at::Tensor GetFOV() const
    {
        if (!mwir_antenna_)
        {
            throw std::runtime_error("Antenna ownership has been transferred.");
        }

        glm::vec2 fov = mwir_antenna_->GetFOV();
        return at::tensor({fov.x, fov.y}, torch::kFloat32).view({2});
    }

    at::Tensor GetRayDensity() const
    {
        if (!mwir_antenna_)
        {
            throw std::runtime_error("Antenna ownership has been transferred.");
        }

        float ray_density = mwir_antenna_->GetRayDensity();
        return at::tensor({ray_density}, torch::kFloat32).view({1});
    }

    at::Tensor GetSolidAngle() const
    {
        if (!mwir_antenna_)
        {
            throw std::runtime_error("Antenna ownership has been transferred.");
        }

        float solid_angle = mwir_antenna_->GetSolidAngle();
        return at::tensor({solid_angle}, torch::kFloat32).view({1});
    }

    at::Tensor GetNRays() const
    {
        if (!mwir_antenna_)
        {
            throw std::runtime_error("Antenna ownership has been transferred.");
        }

        int n_rays = mwir_antenna_->GetNRays();
        return at::tensor({n_rays}, torch::kInt32).view({1});
    }



protected:
    friend class Scene;
    std::unique_ptr<MWIR::Antenna> mwir_antenna_;
};

