#pragma once

#include "mwir/scene.hpp"
#include "mwir/pipeline.hpp"

#include <torch/torch.h>

namespace MWIR
{

class Renderer 
{

public:

    Renderer();
    ~Renderer();    
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    Renderer(Renderer&&) = delete;
    Renderer& operator=(Renderer&&) = delete;

    torch::Tensor Render(Scene &scene, std::optional<torch::Tensor> opt_result_tensor = std::nullopt, std::optional<int> seed = std::nullopt);

private:
    torch::Tensor PrepareRender(Scene &scene, std::optional<torch::Tensor> opt_result_tensor, std::optional<int> seed);
    void RenderAntenna(int sender_index);

    Pipeline pipeline;
    CUstream stream;

    Params params;
    CUdeviceptr d_params = 0;
};

}