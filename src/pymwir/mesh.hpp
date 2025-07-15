#pragma once

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mwir/include/mesh.hpp"

namespace py = pybind11;

class Mesh
{

public:

    Mesh(torch::Tensor &vertices)
    {
        if (vertices.sizes().size() != 2 || vertices.sizes()[1] != 3 || vertices.sizes()[0] < 1)
        {
            throw std::invalid_argument("vertices must be a tensor of shape [N, 3]");
        }

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

        if (vertices.sizes() != std::vector<int64_t>{-1, 3})
        {
            throw std::invalid_argument("vertices must be a tensor of shape [N, 3]");
        }

        std::vector<glm::vec3> vertex_list;
        for (int64_t i = 0; i < vertices.size(0); ++i)
        {
            glm::vec3 vertex(vertices[i][0].item<float>(), vertices[i][1].item<float>(), vertices[i][2].item<float>());
            vertex_list.push_back(vertex);
        }
        mwir_mesh_->SetVertices(std::move(vertex_list));
    }

protected:
    friend class Scene;
    std::unique_ptr<MWIR::Mesh> mwir_mesh_;
};

