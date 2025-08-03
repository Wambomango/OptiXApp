#include "pybindmwir/mesh.hpp"

namespace py = pybind11;

Mesh::Mesh()
{ 
    mwir_mesh_ = std::make_unique<MWIR::Mesh>(std::nullopt, std::nullopt);
}

Mesh::Mesh(std::unique_ptr<MWIR::Mesh> &&impl)
{
    if (!impl)
    {
        throw std::invalid_argument("Mesh implementation cannot be null.");
    }

    mwir_mesh_ = std::move(impl);
}

Mesh::Mesh(torch::Tensor &vertices, torch::Tensor &indices)
{
    mwir_mesh_ = std::make_unique<MWIR::Mesh>(vertices, indices);
}

Mesh Mesh::Clone() const
{
    if (!mwir_mesh_)
    {
        throw std::runtime_error("Mesh ownership has been transferred.");
    }

    return Mesh(std::move(std::make_unique<MWIR::Mesh>(mwir_mesh_->Clone())));
}

void Mesh::SetVertices(torch::Tensor vertices)
{
    if (!mwir_mesh_)
    {
        throw std::runtime_error("Mesh ownership has been transferred.");
    }

    mwir_mesh_->SetVertices(vertices);
}

torch::Tensor Mesh::GetVertices() const
{
    if (!mwir_mesh_)
    {
        throw std::runtime_error("Mesh ownership has been transferred");
    }

    return mwir_mesh_->GetVertices();
}

void Mesh::SetIndices(torch::Tensor indices)
{
    if (!mwir_mesh_)
    {
        throw std::runtime_error("Mesh ownership has been transferred.");
    }

    mwir_mesh_->SetIndices(indices);
}

torch::Tensor Mesh::GetIndices() const
{
    if (!mwir_mesh_)
    {
        throw std::runtime_error("Mesh ownership has been transferred.");
    }

    return mwir_mesh_->GetIndices();
}


void init_mesh(py::module_ &m)
{
    py::class_<Mesh, std::shared_ptr<Mesh>>(m, "Mesh")
        .def(py::init<>())
        .def(py::init<torch::Tensor&, torch::Tensor&>())
        .def("Clone", &Mesh::Clone)
        .def("SetVertices", &Mesh::SetVertices)
        .def("GetVertices", &Mesh::GetVertices)
        .def("SetIndices", &Mesh::SetIndices)
        .def("GetIndices", &Mesh::GetIndices);
}