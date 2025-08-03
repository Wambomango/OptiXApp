#include "mwir/mesh.hpp"
#include <spdlog/spdlog.h>

namespace MWIR
{

Mesh::Mesh(std::optional<torch::Tensor> vertices, std::optional<torch::Tensor> indices)
{
    data = std::make_shared<MeshData>();
    SetVertices(vertices);
    SetIndices(indices);
}

Mesh Mesh::Clone() const
{
    return Mesh(data->vertices.clone(), data->indices.clone());
}

void Mesh::SetVertices(std::optional<torch::Tensor> vertices)
{
    if (vertices.has_value())
    {
        if (vertices->dtype() != torch::kFloat)
        {
            throw std::invalid_argument("Vertices tensor must be of type Float");
        }
        if (vertices->dim() != 2 || vertices->size(1) != 3)
        {
            throw std::invalid_argument("Vertices tensor must have shape [N, 3]");
        }

        data->vertices = vertices.value();
    }
    else
    {
        data->vertices = torch::empty({0, 3}, torch::kFloat);
    }
    data->vertices_updated = true;
}

torch::Tensor Mesh::GetVertices() const
{
    return data->vertices;
}

void Mesh::SetIndices(std::optional<torch::Tensor> indices)
{
    if (indices.has_value())
    {
        if (indices->dtype() != torch::kUInt32)
        {
            throw std::invalid_argument("Indices tensor must be of type UInt32");
        }
        if (indices->dim() != 2 || indices->size(1) != 3)
        {
            throw std::invalid_argument("Indices tensor must have shape [N, 3]");
        }

        data->indices = indices.value();
    }
    else
    {
        data->indices = torch::empty({0, 3}, torch::kUInt32);
    }
    data->indices_updated = true;
}

torch::Tensor Mesh::GetIndices() const
{
    return data->indices;
}

}