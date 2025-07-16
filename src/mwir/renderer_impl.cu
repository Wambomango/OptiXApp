#include "mwir/renderer_impl.hpp"

#include <optix_function_table_definition.h>


__global__ void PrepareGrid(Params *params, int antenna_index)
{
    params->antenna_index = antenna_index;
    params->grid_x = 0;
    params->grid_y = 0;
}

__global__ void AdvanceGridX(Params *params)
{
    params->grid_y = 0;
    params->grid_x++;
}

__global__ void AdvanceGridY(Params *params)
{
    params->grid_y++;
}

__global__ void MergeResults(Params *params, EField *result)
{
    __shared__ EField shared_result[OPTIX_MAX_GRID_DIM];

    int antenna_index = blockIdx.x;
    int frequency_index = blockIdx.y;
    int row_index = threadIdx.x;
    int row_antenna_frequency_offset = row_index * OPTIX_MAX_GRID_DIM * params->n_receivers * params->signal.n_frequencies + antenna_index * params->signal.n_frequencies + frequency_index;

    EField sum = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for(int y = 0; y < OPTIX_MAX_GRID_DIM; y++)
    {
        int idx = row_antenna_frequency_offset + y * params->n_receivers * params->signal.n_frequencies;
        EField cell = params->result[idx];
        sum.x_re += cell.x_re;
        sum.x_im += cell.x_im;
        sum.y_re += cell.y_re;
        sum.y_im += cell.y_im;
        sum.z_re += cell.z_re;
        sum.z_im += cell.z_im;
    }
    shared_result[row_index] = sum;

    __syncthreads();

    if(row_index == 0)
    {
        sum = {115.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for(int i = 0; i < OPTIX_MAX_GRID_DIM; i++)
        {
            sum.x_re += shared_result[i].x_re;
            sum.x_im += shared_result[i].x_im;
            sum.y_re += shared_result[i].y_re;
            sum.y_im += shared_result[i].y_im;
            sum.z_re += shared_result[i].z_re;
            sum.z_im += shared_result[i].z_im;
        }

        result[antenna_index * params->signal.n_frequencies + frequency_index] = sum;
    }
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
    at::Tensor result_tensor = at::empty({n_receivers, n_frequencies, 3}, at::dtype(at::kComplexFloat).device(at::kCUDA, 0));

    CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void *>(d_results), 0, result_bytes, stream));
    for(int i = 0; i < params.n_senders; i++)
    {
        RenderAntenna(i);
    }
    MergeResults<<<dim3(params.n_receivers, params.signal.n_frequencies, 1), OPTIX_MAX_GRID_DIM, sizeof(EField) * OPTIX_MAX_GRID_DIM, stream>>>(reinterpret_cast<Params *>(d_params), static_cast<EField *>(result_tensor.data_ptr()));
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
        result_bytes = OPTIX_MAX_GRID_DIM * OPTIX_MAX_GRID_DIM * n_receivers * n_frequencies * sizeof(EField);
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_results)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_results), result_bytes));
        params.result = reinterpret_cast<EField *>(d_results);
    }

    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(d_params), &params, sizeof(Params), cudaMemcpyHostToDevice, stream));
}

void RendererImpl::RenderAntenna(int sender_index)
{
    glm::ivec2 n_rays = {params.h_senders[sender_index].n_rays.x, params.h_senders[sender_index].n_rays.y};
    SPDLOG_INFO("Rendering antenna {} with {}x{}={} rays", sender_index, n_rays.x, n_rays.y, n_rays.x * n_rays.y);

    int grid_x = std::ceil(float(n_rays.x) / OPTIX_MAX_GRID_DIM);
    int grid_y = std::ceil(float(n_rays.y) / OPTIX_MAX_GRID_DIM);

    PrepareGrid<<<1, 1, 0, stream>>>(reinterpret_cast<Params *>(d_params), sender_index);

    for(int x = 0; x < grid_x; x++)
    {
        int n_x = std::min(n_rays.x - x * OPTIX_MAX_GRID_DIM, OPTIX_MAX_GRID_DIM);
        for(int y = 0; y < grid_y; y++)
        {
            int n_y = std::min(n_rays.y - y * OPTIX_MAX_GRID_DIM, OPTIX_MAX_GRID_DIM);
            OPTIX_CHECK(optixLaunch(forward_pipeline.pipeline->Handle(), stream, d_params, sizeof(Params), &forward_pipeline.sbt, n_x, n_y, 1));

            AdvanceGridY<<<1, 1, 0, stream>>>(reinterpret_cast<Params *>(d_params));
        }
        AdvanceGridX<<<1, 1, 0, stream>>>(reinterpret_cast<Params *>(d_params));
    }
}


}