#include "mwir/mesh.hpp"
#include <spdlog/spdlog.h>

namespace MWIR
{

Mesh::Mesh(std::optional<torch::Tensor> vertices)
{
    SetVertices(vertices);
}

Mesh Mesh::Clone() const
{
    return Mesh(vertices.clone());
}

void Mesh::SetVertices(std::optional<torch::Tensor> vertices)
{
    if (vertices.has_value())
    {
        this->vertices = vertices.value();
    }
    else
    {
        this->vertices = torch::empty({0, 3}, torch::kFloat);
    }
}

torch::Tensor Mesh::GetVertices() const
{
    return vertices;
}

}