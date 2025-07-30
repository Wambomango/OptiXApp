#pragma once

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mwir/include/mesh.hpp"


namespace py = pybind11;

class Mesh
{

public:

    Mesh()
    { 
        mwir_mesh_ = std::make_unique<MWIR::Mesh>();
    }

    Mesh(MWIR::Mesh &&mwir_mesh)
    {
        mwir_mesh_ = std::make_unique<MWIR::Mesh>(std::move(mwir_mesh));
    }

    Mesh(torch::Tensor &vertices)
    {
        std::vector<glm::vec3> vertex_list;
        for (int64_t i = 0; i < vertices.size(0); ++i)
        {
            glm::vec3 vertex(vertices[i][0].item<float>(), vertices[i][1].item<float>(), vertices[i][2].item<float>());
            vertex_list.push_back(vertex);
        }
        mwir_mesh_ = std::make_unique<MWIR::Mesh>(std::move(vertex_list));
    }

    ~Mesh()
    {
    }

    void SetVertices(torch::Tensor &vertices)
    {
        if (!mwir_mesh_)
        {
            throw std::runtime_error("Mesh ownership has been transferred.");
        }

        std::vector<glm::vec3> vertex_list;
        for (int64_t i = 0; i < vertices.size(0); ++i)
        {
            glm::vec3 vertex(vertices[i][0].item<float>(), vertices[i][1].item<float>(), vertices[i][2].item<float>());
            vertex_list.push_back(vertex);
        }
        mwir_mesh_->SetVertices(std::move(vertex_list));
    }

    torch::Tensor GetVertices() const
    {
        if (!mwir_mesh_)
        {
            throw std::runtime_error("Mesh ownership has been transferred");
        }

        const auto& vertices = mwir_mesh_->GetVertices();
        torch::Tensor result = torch::empty({static_cast<int64_t>(vertices.size()), 3}, torch::kFloat32);
        for (size_t i = 0; i < vertices.size(); ++i)
        {
            result[i][0] = vertices[i].x;
            result[i][1] = vertices[i].y;
            result[i][2] = vertices[i].z;
        }
        return result;
    }


protected:
    friend class Scene;
    std::unique_ptr<MWIR::Mesh> mwir_mesh_;
};

