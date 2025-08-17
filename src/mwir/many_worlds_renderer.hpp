#pragma once

#include "mwir/scene.hpp"
#include "mwir/many_worlds_pipeline.hpp"
#include "mwir/many_worlds.hpp"

#include <torch/torch.h>

namespace MWIR
{

class ManyWorldsRenderer 
{

public:

    ManyWorldsRenderer();
    ~ManyWorldsRenderer();    
    ManyWorldsRenderer(const ManyWorldsRenderer&) = delete;
    ManyWorldsRenderer& operator=(const ManyWorldsRenderer&) = delete;
    ManyWorldsRenderer(ManyWorldsRenderer&&) = delete;
    ManyWorldsRenderer& operator=(ManyWorldsRenderer&&) = delete;

    torch::Tensor Forward(Scene &scene, ManyWorlds &many_worlds, std::optional<torch::Tensor> result_tensor = std::nullopt, std::optional<int> seed = std::nullopt);
    std::pair<torch::Tensor, torch::Tensor> Backward(Scene &scene, ManyWorlds &many_worlds, torch::Tensor &output_gradient, std::optional<torch::Tensor> opt_occupancy_gradient, std::optional<torch::Tensor> opt_normal_gradient, std::optional<int> seed);

private:
    torch::Tensor PrepareForward(Scene &scene, ManyWorlds &many_worlds, std::optional<torch::Tensor> result_tensor, std::optional<int> seed);
    std::pair<torch::Tensor, torch::Tensor> PrepareBackward(Scene &scene, ManyWorlds &many_worlds, torch::Tensor &output_gradient, std::optional<torch::Tensor> opt_occupancy_gradient, std::optional<torch::Tensor> opt_normal_gradient, std::optional<int> seed);
    void CheckOutputGradient(torch::Tensor &output_gradient);
    void RenderAntenna(int sender_index);

    ManyWorldsPipeline many_worlds_pipeline;
    CUstream stream;

    Params params;
    CUdeviceptr d_params = 0;

    int n_receivers = 0;
    int n_samples = 0;
    int result_bytes = 0;
    CUdeviceptr d_results = 0;
};

}