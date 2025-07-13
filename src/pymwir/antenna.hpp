#pragma once

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mwir/include/antenna.hpp"

namespace py = pybind11;

class Antenna
{

public:
    Antenna(at::Tensor &position, at::Tensor &orientation, at::Tensor &fov)
    {
        if (position.sizes() != std::vector<int64_t>{3})
        {
            throw std::invalid_argument("position must be a tensor of shape [3]");
        }

        if (orientation.sizes() != std::vector<int64_t>{3, 3})
        {
            throw std::invalid_argument("orientation must be a tensor of shape [3, 3]");
        }

        if (fov.sizes() != std::vector<int64_t>{2})
        {
            throw std::invalid_argument("fov must be a tensor of shape [2]");
        }
    
        mwir_antenna_ = std::make_unique<MWIR::Antenna>(
            glm::vec3(position[0].item<float>(), position[1].item<float>(), position[2].item<float>()),
            glm::mat3(orientation[0][0].item<float>(), orientation[0][1].item<float>(), orientation[0][2].item<float>(),
                      orientation[1][0].item<float>(), orientation[1][1].item<float>(), orientation[1][2].item<float>(),
                      orientation[2][0].item<float>(), orientation[2][1].item<float>(), orientation[2][2].item<float>()),
            glm::vec2(fov[0].item<float>(), fov[1].item<float>()));
    }

    ~Antenna()
    {
    }

    void SetPosition(const at::Tensor &position)
    {
        if (position.sizes() != std::vector<int64_t>{3})
        {
            throw std::invalid_argument("position must be a tensor of shape [3]");
        }
        
        glm::vec3 pos = {position[0].item<float>(), position[1].item<float>(), position[2].item<float>()};
        mwir_antenna_->SetPosition(pos);
    } 

    void SetOrientation(const at::Tensor &orientation)
    {
        if (orientation.sizes() != std::vector<int64_t>{3, 3})
        {
            throw std::invalid_argument("orientation must be a tensor of shape [3, 3]");
        }

        glm::mat3 orient;
        for (size_t i = 0; i < 9; ++i)
        {
            orient[i / 3][i % 3] = orientation[i / 3][i % 3].item<float>();
        }
        mwir_antenna_->SetOrientation(orient);
    }

    void SetFOV(const at::Tensor &fov)
    {
        if (fov.sizes() != std::vector<int64_t>{2})
        {
            throw std::invalid_argument("fov must be a tensor of shape [2]");
        }

        glm::vec2 fov_vec = {fov[0].item<float>(), fov[1].item<float>()};
        mwir_antenna_->SetFOV(fov_vec);
    }

    at::Tensor GetPosition() const
    {
        glm::vec3 pos = mwir_antenna_->GetPosition();
        return at::tensor({pos.x, pos.y, pos.z}, torch::kFloat32).view({3});
    }

    at::Tensor GetOrientation() const
    {
        glm::mat3 orient = mwir_antenna_->GetOrientation();
        return at::tensor({orient[0][0], orient[0][1], orient[0][2],
                            orient[1][0], orient[1][1], orient[1][2],
                            orient[2][0], orient[2][1], orient[2][2]}, torch::kFloat32).view({3, 3});
    }

    at::Tensor GetFOV() const
    {
        glm::vec2 fov = mwir_antenna_->GetFOV();
        return at::tensor({fov.x, fov.y}, torch::kFloat32).view({2});
    }

protected:
    friend class Scene;
    std::unique_ptr<MWIR::Antenna> mwir_antenna_;
};

