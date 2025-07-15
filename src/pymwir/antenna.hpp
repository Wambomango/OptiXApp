#pragma once

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mwir/include/antenna.hpp"

namespace py = pybind11;

class Antenna
{

public:
    Antenna(at::Tensor &position, at::Tensor &euler, at::Tensor &fov, at::Tensor &ray_density)
    {
        if (position.sizes() != std::vector<int64_t>{3})
        {
            throw std::invalid_argument("position must be a tensor of shape [3]");
        }

        if (euler.sizes() != std::vector<int64_t>{3})
        {
            throw std::invalid_argument("euler must be a tensor of shape [3]");
        }

        if (fov.sizes() != std::vector<int64_t>{2})
        {
            throw std::invalid_argument("fov must be a tensor of shape [2]");
        }

        if (ray_density.sizes() != std::vector<int64_t>{1})
        {
            throw std::invalid_argument("ray_density must be a tensor of shape [1]");
        }

        mwir_antenna_ = std::make_unique<MWIR::Antenna>(
            glm::vec3(position[0].item<float>(), position[1].item<float>(), position[2].item<float>()),
            glm::vec3(euler[0].item<float>(), euler[1].item<float>(), euler[2].item<float>()),
            glm::vec2(fov[0].item<float>(), fov[1].item<float>()), ray_density[0].item<float>());
    }

    Antenna(py::tuple position, py::tuple euler, py::tuple fov, float ray_density)
    {
        if (position.size() != 3)
        {
            throw std::invalid_argument("position must be a tuple of length 3");
        }

        if (euler.size() != 3)
        {
            throw std::invalid_argument("euler must be a tuple of length 3");
        }

        if (fov.size() != 2)
        {
            throw std::invalid_argument("fov must be a tuple of length 2");
        }

        mwir_antenna_ = std::make_unique<MWIR::Antenna>(
            glm::vec3(position[0].cast<float>(), position[1].cast<float>(), position[2].cast<float>()),
            glm::vec3(euler[0].cast<float>(), euler[1].cast<float>(), euler[2].cast<float>()),
            glm::vec2(fov[0].cast<float>(), fov[1].cast<float>()), ray_density);
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

        if (position.sizes() != std::vector<int64_t>{3})
        {
            throw std::invalid_argument("position must be a tensor of shape [3]");
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

        if (euler.sizes() != std::vector<int64_t>{3})
        {
            throw std::invalid_argument("euler must be a tensor of shape [3]");
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

        if (fov.sizes() != std::vector<int64_t>{2})
        {
            throw std::invalid_argument("fov must be a tensor of shape [2]");
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

        if (ray_density.sizes() != std::vector<int64_t>{1})
        {
            throw std::invalid_argument("ray_density must be a tensor of shape [1]");
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

        glm::ivec2 n_rays = mwir_antenna_->GetNRays();
        return at::tensor({n_rays.x, n_rays.y}, torch::kInt32).view({2});
    }



protected:
    friend class Scene;
    std::unique_ptr<MWIR::Antenna> mwir_antenna_;
};

