#include "mwir/many_worlds_renderer.hpp"
#include "mwir/kernels.hpp"

namespace MWIR
{

ManyWorldsRenderer::ManyWorldsRenderer() : many_worlds_pipeline()
{    
    OptiX::Context &ctx = Context::GetInstance();
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_params), sizeof(Params)));
}

ManyWorldsRenderer::~ManyWorldsRenderer()
{
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_params)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_results)));
}

torch::Tensor ManyWorldsRenderer::Forward(Scene &scene, ManyWorlds &many_worlds, std::optional<torch::Tensor> opt_result_tensor, std::optional<int> seed)
{
    PrepareRendering(scene, many_worlds, seed);    
    torch::Tensor result_tensor = AllocateResultTensor(opt_result_tensor);

    for(int i = 0; i < params.scene.n_senders; i++)
    {
        RenderAntenna(i);
    }

    MergeResults<<<dim3(params.scene.n_receivers, params.scene.signal.n_samples, 1), OPTIX_MAX_GRID_DIM, sizeof(complex3) * OPTIX_MAX_GRID_DIM, stream>>>(reinterpret_cast<Params *>(d_params), static_cast<complex3 *>(result_tensor.data_ptr()));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    SPDLOG_INFO("Finished rendering");
    return result_tensor;
}

void ManyWorldsRenderer::Backward(Scene &scene, ManyWorlds &many_worlds, torch::Tensor &grad_output, std::optional<int> seed)
{
}

void ManyWorldsRenderer::PrepareRendering(Scene &scene, ManyWorlds &many_worlds, std::optional<int> seed)
{
    scene.PrepareRendering(params, stream);
    many_worlds.PrepareRendering(params, stream);

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
}

torch::Tensor ManyWorldsRenderer::AllocateResultTensor(std::optional<torch::Tensor> opt_result_tensor)
{
    torch::Tensor result_tensor;
    if(opt_result_tensor.has_value())
    {
        result_tensor = opt_result_tensor.value();
         if(result_tensor.device().type() != torch::kCUDA)
        { 
            throw std::runtime_error("Result tensor must be on CUDA device");
        }
        if(result_tensor.dtype() != torch::kComplexFloat)
        {           
            throw std::runtime_error("Result tensor must have dtype torch::kComplexFloat");
        }
        if(result_tensor.dim() != 3 || result_tensor.size(2) != 3)
        {            
            throw std::runtime_error("Result tensor must have shape [n_receivers, n_samples, 3]");
        }
        if(result_tensor.size(0) != params.scene.n_receivers || result_tensor.size(1) != params.scene.signal.n_samples)
        {
            throw std::runtime_error("Result tensor does not match scene parameters: expected [" + std::to_string(params.scene.n_receivers) + ", " + std::to_string(params.scene.signal.n_samples) + ", 3], but got [" + std::to_string(result_tensor.size(0)) + ", " + std::to_string(result_tensor.size(1)) + ", 3]");
        }
    }
    else
    {
        result_tensor = torch::empty({params.scene.n_receivers, params.scene.signal.n_samples, 3}, torch::dtype(torch::kComplexFloat).device(torch::kCUDA, 0));
    }
    return result_tensor;
}

void ManyWorldsRenderer::RenderAntenna(int sender_index)
{
    SPDLOG_INFO("Rendering antenna {} with {} rays", sender_index, params.scene.h_senders[sender_index].n_rays);
    SetAntenna<<<1, 1, 0, stream>>>(reinterpret_cast<Params *>(d_params), sender_index);
    OPTIX_CHECK(optixLaunch(many_worlds_pipeline.pipeline->Handle(), stream, d_params, sizeof(Params), &many_worlds_pipeline.sbt, OPTIX_MAX_GRID_DIM, OPTIX_MAX_GRID_DIM, 1));
}


}