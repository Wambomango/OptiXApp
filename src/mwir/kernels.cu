#include "mwir/kernels.hpp"

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
    int row_antenna_frequency_offset = row_index * OPTIX_MAX_GRID_DIM * params->scene.n_receivers * params->scene.signal.n_samples + antenna_index * params->scene.signal.n_samples + frequency_index;

    complex3 sum = make_complex3(0.0f);
    for(int y = 0; y < OPTIX_MAX_GRID_DIM; y++)
    {
        int idx = row_antenna_frequency_offset + y * params->scene.n_receivers * params->scene.signal.n_samples;
        complex3 cell = params->scene.result[idx];
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