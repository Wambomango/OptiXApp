#include "mwir/forward_renderer.hpp"
#include "mwir/kernels.hpp"

namespace MWIR
{

ForwardRenderer::ForwardRenderer() : forward_pipeline()
{    
    OptiX::Context &ctx = Context::GetInstance();
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_params), sizeof(Params)));
}

ForwardRenderer::~ForwardRenderer()
{
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_params)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_results)));
}

torch::Tensor ForwardRenderer::Render(Scene &scene, std::optional<torch::Tensor> opt_result_tensor)
{   
    UpdateParams(scene);
    torch::Tensor result_tensor = AllocateResultTensor(opt_result_tensor);

    CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void *>(d_results), 0, result_bytes, stream));
    for(int i = 0; i < params.scene.n_senders; i++)
    {
        RenderAntenna(i);
    }

    MergeResults<<<dim3(params.scene.n_receivers, params.scene.signal.n_samples, 1), OPTIX_MAX_GRID_DIM, sizeof(complex3) * OPTIX_MAX_GRID_DIM, stream>>>(reinterpret_cast<Params *>(d_params), static_cast<complex3 *>(result_tensor.data_ptr()));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    SPDLOG_INFO("Finished rendering");
    return result_tensor;
}

void ForwardRenderer::UpdateParams(Scene &scene)
{
    params.scene = scene.GetParams();
    int new_n_receivers = params.scene.n_receivers;
    int new_n_samples = params.scene.signal.n_samples;
    if(new_n_receivers != n_receivers || new_n_samples != n_samples)
    {
        n_receivers = new_n_receivers;
        n_samples = new_n_samples;
        result_bytes = OPTIX_MAX_GRID_DIM * OPTIX_MAX_GRID_DIM * n_receivers * n_samples * sizeof(complex3);
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_results)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_results), result_bytes));
        params.result = reinterpret_cast<complex3 *>(d_results);
    }

    srand(static_cast<unsigned int>(time(nullptr)));
    params.seed = rand() % 1000;
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(d_params), &params, sizeof(Params), cudaMemcpyHostToDevice, stream));
}

torch::Tensor ForwardRenderer::AllocateResultTensor(std::optional<torch::Tensor> opt_result_tensor)
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

void ForwardRenderer::RenderAntenna(int sender_index)
{
    SPDLOG_INFO("Rendering antenna {} with {} rays", sender_index, params.scene.h_senders[sender_index].n_rays);
    SetAntenna<<<1, 1, 0, stream>>>(reinterpret_cast<Params *>(d_params), sender_index);
    OPTIX_CHECK(optixLaunch(forward_pipeline.pipeline->Handle(), stream, d_params, sizeof(Params), &forward_pipeline.sbt, OPTIX_MAX_GRID_DIM, OPTIX_MAX_GRID_DIM, 1));
}


}