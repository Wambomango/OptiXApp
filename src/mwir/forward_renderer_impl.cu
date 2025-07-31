#include "mwir/forward_renderer_impl.hpp"

#include <optix_function_table_definition.h>


__global__ void SetAntenna(Params *params, int antenna_index)
{
    params->antenna_index = antenna_index;
}

__global__ void MergeResults(Params *params, complex3 *result)
{
    __shared__ complex3 shared_result[OPTIX_MAX_GRID_DIM];

    int antenna_index = blockIdx.x;
    int frequency_index = blockIdx.y;
    int row_index = threadIdx.x;
    int row_antenna_frequency_offset = row_index * OPTIX_MAX_GRID_DIM * params->n_receivers * params->signal.n_samples + antenna_index * params->signal.n_samples + frequency_index;

    complex3 sum = make_complex3(0.0f);
    for(int y = 0; y < OPTIX_MAX_GRID_DIM; y++)
    {
        int idx = row_antenna_frequency_offset + y * params->n_receivers * params->signal.n_samples;
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

        result[antenna_index * params->signal.n_samples + frequency_index] = sum;
    }
}   


namespace MWIR
{

ForwardRendererImpl::ForwardRendererImpl(SceneImpl &&scene) : forward_pipeline(), scene(std::move(scene))
{    
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_params), sizeof(Params)));
    UpdateParams();
}

ForwardRendererImpl::~ForwardRendererImpl()
{
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_params)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_results)));
}

void ForwardRendererImpl::SetScene(SceneImpl &&scene)
{
    this->scene = std::move(scene);
    UpdateParams();
}

SceneImpl ForwardRendererImpl::GetScene()
{
    SceneImpl tmp = std::move(scene);
    scene = SceneImpl();
    return tmp;
}

at::Tensor ForwardRendererImpl::Render()
{   
    at::Tensor result_tensor = at::empty({n_receivers, n_samples, 3}, at::dtype(at::kComplexFloat).device(at::kCUDA, 0));

    CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void *>(d_results), 0, result_bytes, stream));
    for(int i = 0; i < params.n_senders; i++)
    {
        RenderAntenna(i);
    }
    MergeResults<<<dim3(params.n_receivers, params.signal.n_samples, 1), OPTIX_MAX_GRID_DIM, sizeof(complex3) * OPTIX_MAX_GRID_DIM, stream>>>(reinterpret_cast<Params *>(d_params), static_cast<complex3 *>(result_tensor.data_ptr()));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    SPDLOG_INFO("Finished rendering");
    return result_tensor;
}

void ForwardRendererImpl::UpdateParams()
{
    scene.UpdateParams(params);

    int new_n_receivers = params.n_receivers;
    int new_n_samples = params.signal.n_samples;

    if(new_n_receivers != n_receivers || new_n_samples != n_samples)
    {
        n_receivers = new_n_receivers;
        n_samples = new_n_samples;
        result_bytes = OPTIX_MAX_GRID_DIM * OPTIX_MAX_GRID_DIM * n_receivers * n_samples * sizeof(complex3);
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_results)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_results), result_bytes));
        params.result = reinterpret_cast<complex3 *>(d_results);
    }
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(d_params), &params, sizeof(Params), cudaMemcpyHostToDevice, stream));
}

void ForwardRendererImpl::RenderAntenna(int sender_index)
{
    SPDLOG_INFO("Rendering antenna {} with {} rays", sender_index, params.h_senders[sender_index].n_rays);
    SetAntenna<<<1, 1, 0, stream>>>(reinterpret_cast<Params *>(d_params), sender_index);
    OPTIX_CHECK(optixLaunch(forward_pipeline.pipeline->Handle(), stream, d_params, sizeof(Params), &forward_pipeline.sbt, OPTIX_MAX_GRID_DIM, OPTIX_MAX_GRID_DIM, 1));
}


}