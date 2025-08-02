#pragma once

#include <torch/torch.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mwir/antenna.hpp"

namespace py = pybind11;

class Antenna
{

public:

    Antenna()
    {
        mwir_antenna_ = std::make_unique<MWIR::Antenna>(std::nullopt, std::nullopt, std::nullopt, std::nullopt);
    }

    Antenna(torch::Tensor &position, torch::Tensor &euler, torch::Tensor &fov, torch::Tensor &ray_density)
    {
        mwir_antenna_ = std::make_unique<MWIR::Antenna>(
            glm::vec3(position[0].item<float>(), position[1].item<float>(), position[2].item<float>()),
            glm::vec3(euler[0].item<float>(), euler[1].item<float>(), euler[2].item<float>()),
            glm::vec2(fov[0].item<float>(), fov[1].item<float>()), ray_density[0].item<float>());
    }
 
    Antenna(std::unique_ptr<MWIR::Antenna> &&impl)
    {
        if (!impl)
        {
            throw std::invalid_argument("Antenna implementation cannot be null.");
        }

        mwir_antenna_ = std::move(impl);
    }

    Antenna Clone() const
    {
        if (!mwir_antenna_)
        {
            throw std::runtime_error("Antenna ownership has been transferred.");
        }

        return Antenna(std::move(std::make_unique<MWIR::Antenna>(mwir_antenna_->Clone())));
    }

    void SetPosition(const torch::Tensor &position)
    {
        if (!mwir_antenna_)
        {
            throw std::runtime_error("Antenna ownership has been transferred.");
        }

        glm::vec3 pos = {position[0].item<float>(), position[1].item<float>(), position[2].item<float>()};
        mwir_antenna_->SetPosition(pos);
    } 

    void SetOrientation(const torch::Tensor &euler)
    {
        if (!mwir_antenna_)
        {
            throw std::runtime_error("Antenna ownership has been transferred.");
        }

        glm::vec3 euler_vec = {euler[0].item<float>(), euler[1].item<float>(), euler[2].item<float>()};
        mwir_antenna_->SetOrientation(euler_vec);
    }

    void SetFOV(const torch::Tensor &fov)
    {
        if (!mwir_antenna_)
        {
            throw std::runtime_error("Antenna ownership has been transferred.");
        }

        glm::vec2 fov_vec = {fov[0].item<float>(), fov[1].item<float>()};
        mwir_antenna_->SetFOV(fov_vec);
    }

    void SetRayDensity(const torch::Tensor &ray_density)
    {
        if (!mwir_antenna_)
        {
            throw std::runtime_error("Antenna ownership has been transferred.");
        }

        mwir_antenna_->SetRayDensity(ray_density[0].item<float>());
    }

    torch::Tensor GetPosition() const
    {
        if (!mwir_antenna_)
        {
            throw std::runtime_error("Antenna ownership has been transferred.");
        }

        glm::vec3 pos = mwir_antenna_->GetPosition();
        torch::Tensor result = torch::tensor({pos.x, pos.y, pos.z}, torch::kFloat32);
        result = result.view({3});
        return result;
    }

    torch::Tensor GetOrientation() const
    {
        if (!mwir_antenna_)
        {
            throw std::runtime_error("Antenna ownership has been transferred.");
        }

        glm::vec3 euler = mwir_antenna_->GetOrientation();
        torch::Tensor result = torch::tensor({euler.x, euler.y, euler.z}, torch::kFloat32);
        result = result.view({3});
        return result;
    }

    torch::Tensor GetFOV() const
    {
        if (!mwir_antenna_)
        {
            throw std::runtime_error("Antenna ownership has been transferred.");
        }

        glm::vec2 fov = mwir_antenna_->GetFOV();
        torch::Tensor result = torch::tensor({fov.x, fov.y}, torch::kFloat32);
        result = result.view({2});
        return result;
    }

    torch::Tensor GetRayDensity() const
    {
        if (!mwir_antenna_)
        {
            throw std::runtime_error("Antenna ownership has been transferred.");
        }

        float ray_density = mwir_antenna_->GetRayDensity();
        torch::Tensor result = torch::tensor({ray_density}, torch::kFloat32);
        result = result.view({1});
        return result;
    }

    torch::Tensor GetSolidAngle() const
    {
        if (!mwir_antenna_)
        {
            throw std::runtime_error("Antenna ownership has been transferred.");
        }

        float solid_angle = mwir_antenna_->GetSolidAngle();
        torch::Tensor result = torch::tensor({solid_angle}, torch::kFloat32);
        result = result.view({1});
        return result;      
    }

    torch::Tensor GetNRays() const
    {
        if (!mwir_antenna_)
        {
            throw std::runtime_error("Antenna ownership has been transferred.");
        }

        int n_rays = mwir_antenna_->GetNRays();
        torch::Tensor result = torch::tensor({static_cast<float>(n_rays)}, torch::kFloat32);
        result = result.view({1});
        return result;
    }

protected:
    friend class Scene;
    std::unique_ptr<MWIR::Antenna> mwir_antenna_;
};

