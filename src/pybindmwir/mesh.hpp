#pragma once

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mwir/mesh.hpp"


namespace py = pybind11;

class Mesh
{

public:

    Mesh()
    { 
        mwir_mesh_ = std::make_unique<MWIR::Mesh>(std::nullopt);
    }
   
    Mesh(std::unique_ptr<MWIR::Mesh> &&impl)
    {
        if (!impl)
        {
            throw std::invalid_argument("Mesh implementation cannot be null.");
        }

        mwir_mesh_ = std::move(impl);
    }

    Mesh(torch::Tensor &vertices)
    {
        mwir_mesh_ = std::make_unique<MWIR::Mesh>(vertices);
    }

    Mesh Clone() const
    {
        if (!mwir_mesh_)
        {
            throw std::runtime_error("Mesh ownership has been transferred.");
        }

        return Mesh(std::move(std::make_unique<MWIR::Mesh>(mwir_mesh_->Clone())));
    }

    void SetVertices(torch::Tensor &vertices)
    {
        if (!mwir_mesh_)
        {
            throw std::runtime_error("Mesh ownership has been transferred.");
        }

        mwir_mesh_->SetVertices(vertices);
    }

    torch::Tensor GetVertices() const
    {
        if (!mwir_mesh_)
        {
            throw std::runtime_error("Mesh ownership has been transferred");
        }

        return mwir_mesh_->GetVertices();
    }


protected:
    friend class Scene;
    std::unique_ptr<MWIR::Mesh> mwir_mesh_;
};

