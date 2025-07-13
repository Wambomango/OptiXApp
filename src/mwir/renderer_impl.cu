#include "mwir/renderer_impl.hpp"

#include <optix_function_table_definition.h>

__global__ void AdvanceBatch(CUdeviceptr d_params)
{
    Params *params = reinterpret_cast<Params *>(d_params);
    params->batch_number++;
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
}

void RendererImpl::Render(int n_rays, int batch_size)
{    
    if (n_rays <= 0)
    {
        throw std::invalid_argument("Number of rays must be greater than zero.");
    }
    n_rays = ((n_rays / batch_size) + 1) * batch_size;

    scene.UpdateParams(params);
    CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void *>(d_results), 0, n_receivers * n_frequencies * sizeof(EField), stream));

    while(n_rays > 0)
    {
        n_rays -= batch_size;
        OPTIX_CHECK(optixLaunch(forward_pipeline.pipeline->Handle(), stream, d_params, sizeof(Params), &forward_pipeline.sbt, batch_size, params.n_senders, 1));
        AdvanceBatch<<<1, 1, 0, stream>>>(d_params);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpyAsync(&results, reinterpret_cast<void *>(d_results), n_receivers * n_frequencies * sizeof(EField), cudaMemcpyDeviceToHost, stream));
}

void RendererImpl::UpdateParams()
{
    params.batch_number = 0;

    scene.UpdateParams(params);
    int new_n_receivers = params.n_receivers;
    int new_n_frequencies = params.signal.n_frequencies;

    if(new_n_receivers != n_receivers || new_n_frequencies != n_frequencies)
    {
        n_receivers = new_n_receivers;
        n_frequencies = new_n_frequencies;        
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_results)));
        size_t results_size = n_receivers * n_frequencies * sizeof(EField);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_results), results_size));
        params.result = reinterpret_cast<EField *>(d_results);

        if (results)
        {
            delete[] results;
        }
        results = new EField[n_receivers * n_frequencies];
    }

    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(d_params), &params, sizeof(Params), cudaMemcpyHostToDevice, stream));
}

}