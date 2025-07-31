#pragma once

#include "mwir/scene_impl.hpp"
#include "mwir/forward_pipeline.hpp"

#include <torch/torch.h>

namespace MWIR
{

class ForwardRendererImpl 
{

public:

    ForwardRendererImpl(SceneImpl &&scene);
    ~ForwardRendererImpl();    
    ForwardRendererImpl(const ForwardRendererImpl&) = delete;
    ForwardRendererImpl& operator=(const ForwardRendererImpl&) = delete;
    ForwardRendererImpl(ForwardRendererImpl&&) = default;
    ForwardRendererImpl& operator=(ForwardRendererImpl&&) = default;

    void SetScene(SceneImpl &&scene);
    SceneImpl GetScene();
    at::Tensor Render();

private:
    void UpdateParams();
    void RenderAntenna(int sender_index);

    ForwardPipeline forward_pipeline;
    SceneImpl scene;
    CUstream stream;

    Params params;
    CUdeviceptr d_params = 0;

    int n_receivers = 0;
    int n_samples = 0;
    int result_bytes = 0;
    CUdeviceptr d_results = 0;
};

}