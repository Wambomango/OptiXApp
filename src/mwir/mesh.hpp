#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>
#include <atomic>
#include <optional>

namespace MWIR
{

class Mesh
{
    public:
        Mesh(std::optional<torch::Tensor> vertices, std::optional<torch::Tensor> indices);
        Mesh Clone() const;

        void SetVertices(std::optional<torch::Tensor> vertices);
        torch::Tensor GetVertices() const;

        void SetIndices(std::optional<torch::Tensor> indices);
        torch::Tensor GetIndices() const;
 
    protected:
        friend class Scene;

        struct MeshData
        {
            bool vertices_updated = true;
            bool indices_updated = true;
            torch::Tensor vertices;
            torch::Tensor indices;
        };

        std::shared_ptr<MeshData> data;
};


}