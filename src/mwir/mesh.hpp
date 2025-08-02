#pragma once

#include <glm/glm.hpp>
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
        Mesh(std::optional<torch::Tensor> vertices);
        Mesh Clone() const;

        void SetVertices(std::optional<torch::Tensor> vertices);
        torch::Tensor GetVertices() const;
 
    private:
        torch::Tensor vertices;
};


}