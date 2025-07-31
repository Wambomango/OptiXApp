#pragma once

#include "mwir/scene_impl.hpp"
#include "mwir/inverse_pipeline.hpp"

#include <torch/torch.h>

namespace MWIR
{

class InverseRendererImpl 
{

public:

    InverseRendererImpl();
    ~InverseRendererImpl();    
    InverseRendererImpl(const InverseRendererImpl&) = delete;
    InverseRendererImpl& operator=(const InverseRendererImpl&) = delete;
    InverseRendererImpl(InverseRendererImpl&&) = default;
    InverseRendererImpl& operator=(InverseRendererImpl&&) = default;

    at::Tensor Render(SceneImpl &scene, std::optional<at::Tensor> result_tensor = std::nullopt);

private:
    void UpdateParams(SceneImpl &scene);
    at::Tensor AllocateResultTensor(std::optional<at::Tensor> result_tensor);
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