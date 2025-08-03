#pragma once

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mwir/mesh.hpp"

class Mesh
{

public:

    Mesh();
    Mesh(std::unique_ptr<MWIR::Mesh> &&impl);
    Mesh(torch::Tensor &vertices, torch::Tensor &indices);
    Mesh Clone() const;

    void SetVertices(torch::Tensor vertices);
    torch::Tensor GetVertices() const;
    void SetIndices(torch::Tensor indices);
    torch::Tensor GetIndices() const;

protected:
    friend class Scene;
    std::unique_ptr<MWIR::Mesh> mwir_mesh_;
};

void init_mesh(pybind11::module_ &m);