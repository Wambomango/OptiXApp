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
    torch::Tensor result_tensor = PrepareForward(scene, many_worlds, opt_result_tensor, seed);

    for(int i = 0; i < params.scene.n_senders; i++)
    {
        RenderAntenna(i);
    }
    
    MergeResults<<<dim3(params.scene.n_receivers, params.scene.signal.n_samples, 1), OPTIX_MAX_GRID_DIM, sizeof(complex3) * OPTIX_MAX_GRID_DIM, stream>>>(reinterpret_cast<Params *>(d_params));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    SPDLOG_INFO("Finished rendering");
    return result_tensor;
}

std::pair<torch::Tensor, torch::Tensor> ManyWorldsRenderer::Backward(Scene &scene, ManyWorlds &many_worlds, torch::Tensor &output_gradient, std::optional<torch::Tensor> opt_occupancy_gradient, std::optional<torch::Tensor> opt_normal_gradient, std::optional<int> seed)
{
    std::pair<torch::Tensor, torch::Tensor> grad_tensors = PrepareBackward(scene, many_worlds, output_gradient, opt_occupancy_gradient, opt_normal_gradient, seed);

    for(int i = 0; i < params.scene.n_senders; i++)
    {
        RenderAntenna(i);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    SPDLOG_INFO("Finished rendering");
    return grad_tensors;
}

torch::Tensor ManyWorldsRenderer::PrepareForward(Scene &scene, ManyWorlds &many_worlds, std::optional<torch::Tensor> opt_result_tensor, std::optional<int> seed)
{
    torch::Tensor result_tensor = scene.PrepareRendering(params, opt_result_tensor, stream);
    many_worlds.PrepareForward(params, stream);
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


std::pair<torch::Tensor, torch::Tensor> ManyWorldsRenderer::PrepareBackward(Scene &scene, ManyWorlds &many_worlds, torch::Tensor &output_gradient, std::optional<torch::Tensor> opt_occupancy_gradient, std::optional<torch::Tensor> opt_normal_gradient, std::optional<int> seed)
{
    scene.PrepareRendering(params, std::nullopt, stream);
    std::pair<torch::Tensor, torch::Tensor> grad_tensors = many_worlds.PrepareBackward(params, output_gradient, opt_occupancy_gradient, opt_normal_gradient, stream);
    CheckOutputGradient(output_gradient);
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
    return grad_tensors;
}

void ManyWorldsRenderer::CheckOutputGradient(torch::Tensor &output_gradient)
{
    if (output_gradient.device().type() != torch::kCUDA)
    {
        throw std::runtime_error("Output gradient must be on CUDA device");
    }
    if (output_gradient.dtype() != torch::kComplexFloat)
    {
        throw std::runtime_error("Output gradient must have dtype torch::kComplexFloat");
    }
    if (output_gradient.dim() != 3 || output_gradient.size(0) != params.scene.n_receivers || output_gradient.size(1) != params.scene.signal.n_samples || output_gradient.size(2) != 3)
    {
        throw std::runtime_error("Output gradient must have shape [n_receivers, n_samples, 3]");
    }
}

void ManyWorldsRenderer::RenderAntenna(int sender_index)
{
    SPDLOG_INFO("Rendering antenna {} with {} rays", sender_index, params.scene.h_senders[sender_index].n_rays);
    SetAntenna<<<1, 1, 0, stream>>>(reinterpret_cast<Params *>(d_params), sender_index);
    OPTIX_CHECK(optixLaunch(many_worlds_pipeline.pipeline->Handle(), stream, d_params, sizeof(Params), &many_worlds_pipeline.sbt, OPTIX_MAX_GRID_DIM, OPTIX_MAX_GRID_DIM, 1));
}


}