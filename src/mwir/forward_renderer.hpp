#pragma once

#include "mwir/scene.hpp"
#include "mwir/forward_pipeline.hpp"

#include <torch/torch.h>

namespace MWIR
{

class ForwardRenderer 
{

public:

    ForwardRenderer();
    ~ForwardRenderer();    
    ForwardRenderer(const ForwardRenderer&) = delete;
    ForwardRenderer& operator=(const ForwardRenderer&) = delete;
    ForwardRenderer(ForwardRenderer&&) = delete;
    ForwardRenderer& operator=(ForwardRenderer&&) = delete;

    torch::Tensor Render(Scene &scene, std::optional<torch::Tensor> result_tensor = std::nullopt);

private:
    void PrepareRendering(Scene &scene);
    torch::Tensor AllocateResultTensor(std::optional<torch::Tensor> result_tensor);
    void RenderAntenna(int sender_index);

    ForwardPipeline forward_pipeline;
    CUstream stream;

    Params params;
    CUdeviceptr d_params = 0;
};

}