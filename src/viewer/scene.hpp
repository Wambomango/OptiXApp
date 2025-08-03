#pragma once

#include <tiny_obj_loader.h>
#include <spdlog/spdlog.h>
#include <filesystem>
#include <glm/glm.hpp>
#include <torch/torch.h>

class Scene
{
    public:

        Scene(std::string path);
        void GetMesh(torch::Tensor &vertices, torch::Tensor &indices);

    private:
        void LoadFromObj(torch::Tensor &vertices, torch::Tensor &indices);
        void LoadFromPt(torch::Tensor &vertices, torch::Tensor &indices);

        std::string path;

        torch::Tensor vertices;
        torch::Tensor indices;

};