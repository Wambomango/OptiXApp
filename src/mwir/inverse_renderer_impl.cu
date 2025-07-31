#include "mwir/inverse_renderer_impl.hpp"

__global__ void INVSetAntenna(Params *params, int antenna_index)
{
    params->antenna_index = antenna_index;
}

__global__ void INVMergeResults(Params *params, complex3 *result)
{
    __shared__ complex3 shared_result[OPTIX_MAX_GRID_DIM];

    int antenna_index = blockIdx.x;
    int frequency_index = blockIdx.y;
    int row_index = threadIdx.x;
    int row_antenna_frequency_offset = row_index * OPTIX_MAX_GRID_DIM * params->scene.n_receivers * params->scene.signal.n_samples + antenna_index * params->scene.signal.n_samples + frequency_index;

    complex3 sum = make_complex3(0.0f);
    for(int y = 0; y < OPTIX_MAX_GRID_DIM; y++)
    {
        int idx = row_antenna_frequency_offset + y * params->scene.n_receivers * params->scene.signal.n_samples;
        complex3 cell = params->result[idx];
        sum += cell;
    }
    shared_result[row_index] = sum;

    __syncthreads();

    if(row_index == 0)
    {
        sum = make_complex3(0.0f);
        for(int i = 0; i < OPTIX_MAX_GRID_DIM; i++)
        {
            sum += shared_result[i];
        }

        result[antenna_index * params->scene.signal.n_samples + frequency_index] = sum;
    }
}   


namespace MWIR
{

InverseRendererImpl::InverseRendererImpl() : inverse_pipeline()
{    
    OptiX::Context &ctx = Context::GetInstance();
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_params), sizeof(Params)));
}

InverseRendererImpl::~InverseRendererImpl()
{
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_params)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_results)));
}

at::Tensor InverseRendererImpl::Render(SceneImpl &scene, std::optional<at::Tensor> opt_result_tensor)
{   
    UpdateParams(scene);
    at::Tensor result_tensor = AllocateResultTensor(opt_result_tensor);

    CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void *>(d_results), 0, result_bytes, stream));
    for(int i = 0; i < params.scene.n_senders; i++)
    {
        RenderAntenna(i);
    }

    INVMergeResults<<<dim3(params.scene.n_receivers, params.scene.signal.n_samples, 1), OPTIX_MAX_GRID_DIM, sizeof(complex3) * OPTIX_MAX_GRID_DIM, stream>>>(reinterpret_cast<Params *>(d_params), static_cast<complex3 *>(result_tensor.data_ptr()));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    SPDLOG_INFO("Finished rendering");
    return result_tensor;
}

void InverseRendererImpl::UpdateParams(SceneImpl &scene)
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

at::Tensor InverseRendererImpl::AllocateResultTensor(std::optional<at::Tensor> opt_result_tensor)
{
    at::Tensor result_tensor;
    if(opt_result_tensor.has_value())
    {
        result_tensor = opt_result_tensor.value();
         if(result_tensor.device().type() != at::kCUDA)
        { 
            throw std::runtime_error("Result tensor must be on CUDA device");
        }
        if(result_tensor.dtype() != at::kComplexFloat)
        {           
            throw std::runtime_error("Result tensor must have dtype at::kComplexFloat");
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
        result_tensor = at::empty({params.scene.n_receivers, params.scene.signal.n_samples, 3}, at::dtype(at::kComplexFloat).device(at::kCUDA, 0));
    }
    return result_tensor;
}

void InverseRendererImpl::RenderAntenna(int sender_index)
{
    SPDLOG_INFO("Rendering antenna {} with {} rays", sender_index, params.scene.h_senders[sender_index].n_rays);
    INVSetAntenna<<<1, 1, 0, stream>>>(reinterpret_cast<Params *>(d_params), sender_index);
    OPTIX_CHECK(optixLaunch(inverse_pipeline.pipeline->Handle(), stream, d_params, sizeof(Params), &inverse_pipeline.sbt, OPTIX_MAX_GRID_DIM, OPTIX_MAX_GRID_DIM, 1));
}


}