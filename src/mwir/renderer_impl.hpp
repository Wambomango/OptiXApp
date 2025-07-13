#pragma once

#include "mwir/scene_impl.hpp"
#include "mwir/forward_pipeline.hpp"


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
    void Render(int n_rays, int batch_size);

private:
    void UpdateParams();

    ForwardPipeline forward_pipeline;
    SceneImpl scene;
    CUstream stream;

    Params params;
    CUdeviceptr d_params = 0;

    EField *results = nullptr;
    CUdeviceptr d_results = 0;
    int n_receivers = 0;
    int n_frequencies = 0;
};

}