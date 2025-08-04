#pragma once

#include "mwir/inverse_pipeline.hpp"

#include "mwir/scene.hpp"
#include "mwir/many_worlds.hpp"

#include <torch/torch.h>

namespace MWIR
{

class InverseRenderer 
{

public:

    InverseRenderer();
    ~InverseRenderer();    
    InverseRenderer(const InverseRenderer&) = delete;
    InverseRenderer& operator=(const InverseRenderer&) = delete;
    InverseRenderer(InverseRenderer&&) = delete;
    InverseRenderer& operator=(InverseRenderer&&) = delete;

    torch::Tensor Render(Scene &scene, ManyWorlds &many_worlds, std::optional<torch::Tensor> result_tensor = std::nullopt);

private:
    void UpdateParams(Scene &scene, ManyWorlds &many_worlds);
    torch::Tensor AllocateResultTensor(std::optional<torch::Tensor> result_tensor);
    void RenderAntenna(int sender_index);

    InversePipeline inverse_pipeline;
    CUstream stream;

    Params params;
    CUdeviceptr d_params = 0;

    int n_receivers = 0;
    int n_samples = 0;
    int result_bytes = 0;
    CUdeviceptr d_results = 0;
};

}