#include "mwir/renderer.hpp"
#include "mwir/kernels.hpp"

namespace MWIR
{

Renderer::Renderer() : pipeline()
{    
    OptiX::Context &ctx = Context::GetInstance();
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_params), sizeof(Params)));
}

Renderer::~Renderer()
{
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_params)));
}

torch::Tensor Renderer::Render(Scene &scene, std::optional<torch::Tensor> opt_result_tensor, std::optional<int> seed)
{   
    torch::Tensor result_tensor = PrepareRender(scene, opt_result_tensor, seed);   

    for(int i = 0; i < params.scene.n_senders; i++)
    {
        RenderAntenna(i);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    MergeResults<<<dim3(params.scene.n_receivers, params.scene.signal.n_samples, 1), OPTIX_MAX_GRID_DIM, sizeof(complex3) * OPTIX_MAX_GRID_DIM, stream>>>(reinterpret_cast<Params *>(d_params));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    SPDLOG_INFO("Finished rendering");
    return result_tensor;
}

torch::Tensor Renderer::PrepareRender(Scene &scene, std::optional<torch::Tensor> opt_result_tensor, std::optional<int> seed)
{   
    torch::Tensor result_tensor = scene.PrepareRendering(params, opt_result_tensor, stream);

    if (seed.has_value())
    {
        params.seed = seed.value();
    }
    else
    {
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
        params.seed = std::rand();
    }
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(d_params), &params, sizeof(Params), cudaMemcpyHostToDevice, stream));
    return result_tensor;
}

void Renderer::RenderAntenna(int sender_index)
{
    SPDLOG_INFO("Rendering antenna {} with {} rays", sender_index, params.scene.h_senders[sender_index].n_rays);

    SetAntenna<<<1, 1, 0, stream>>>(reinterpret_cast<Params *>(d_params), sender_index);
    OPTIX_CHECK(optixLaunch(pipeline.pipeline->Handle(), stream, d_params, sizeof(Params), &pipeline.sbt, OPTIX_MAX_GRID_DIM, OPTIX_MAX_GRID_DIM, 1));
}


}