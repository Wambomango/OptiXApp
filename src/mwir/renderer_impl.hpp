#pragma once

#include "mwir/scene_impl.hpp"
#include "mwir/forward_pipeline.hpp"

#include <torch/torch.h>

namespace MWIR
{

class RendererImpl 
{

public:

    RendererImpl(SceneImpl &&scene);
    ~RendererImpl();    
    RendererImpl(const RendererImpl&) = delete;
    RendererImpl& operator=(const RendererImpl&) = delete;
    RendererImpl(RendererImpl&&) = default;
    RendererImpl& operator=(RendererImpl&&) = default;

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