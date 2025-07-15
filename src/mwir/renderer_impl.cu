#include "mwir/renderer_impl.hpp"

#include <optix_function_table_definition.h>


__global__ void AdvanceAntenna(Params *params)
{
    params->antenna_index++;
}

namespace MWIR
{

RendererImpl::RendererImpl(SceneImpl &&scene) : forward_pipeline(), scene(std::move(scene))
{    
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_params), sizeof(Params)));
    UpdateParams();
}

RendererImpl::~RendererImpl()
{
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_params)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_results)));
}

void RendererImpl::SetScene(SceneImpl &&scene)
{
    this->scene = std::move(scene);
    scene.UpdateParams(params);
}

at::Tensor RendererImpl::Render()
{   
    params.antenna_index = 0;
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(d_params), &params, sizeof(Params), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void *>(d_results), 0, result_bytes, stream));

    for(int i = 0; i < params.n_senders; ++i)
    {
        SPDLOG_INFO("Rendering sender {}/{} with {}/{} rays", i + 1, params.n_senders, params.h_senders[i].n_rays.x, params.h_senders[i].n_rays.y);
        OPTIX_CHECK(optixLaunch(forward_pipeline.pipeline->Handle(), stream, d_params, sizeof(Params), &forward_pipeline.sbt, params.h_senders[i].n_rays.x, params.h_senders[i].n_rays.y, 1));
        AdvanceAntenna<<<1, 1, 0, stream>>>(reinterpret_cast<Params *>(d_params));
    }
    
    at::Tensor result_tensor = at::empty({n_receivers, n_frequencies, 3}, at::dtype(at::kComplexFloat).device(at::kCPU, 0));
    CUDA_CHECK(cudaMemcpyAsync(result_tensor.data_ptr(), reinterpret_cast<void *>(d_results), result_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    SPDLOG_INFO("Finished rendering");
    return result_tensor;
}

void RendererImpl::UpdateParams()
{
    scene.UpdateParams(params);

    int new_n_receivers = params.n_receivers;
    int new_n_frequencies = params.signal.n_frequencies;

    if(new_n_receivers != n_receivers || new_n_frequencies != n_frequencies)
    {
        n_receivers = new_n_receivers;
        n_frequencies = new_n_frequencies;
        result_bytes = n_receivers * n_frequencies * sizeof(EField);
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_results)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_results), result_bytes));
        params.result = reinterpret_cast<EField *>(d_results);
    }

    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(d_params), &params, sizeof(Params), cudaMemcpyHostToDevice, stream));
}

}