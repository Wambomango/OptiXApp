#include "mwir/many_worlds.hpp"
#include <spdlog/spdlog.h>

#include "utils/optix/torch.hpp"
#include "mwir/context.hpp"

using torch::indexing::Slice;

namespace MWIR
{

ManyWorlds::ManyWorlds(std::optional<glm::vec3> min, std::optional<glm::vec3> max, std::optional<float> resolution, std::optional<int> n_samples)
{
    data = std::make_shared<ManyWorldsData>();
    data->min = min.value_or(glm::vec3(-0.1f));
    data->max = max.value_or(glm::vec3(0.1f));
    data->resolution = resolution.value_or(0.001f);
    data->n_samples = n_samples.value_or(1);
    UpdateShape();
}

ManyWorlds ManyWorlds::Clone() const 
{
    ManyWorlds clone(data->min, data->max, data->resolution, data->n_samples);
    clone.data->occupancy = data->occupancy.clone();
    clone.data->normal = data->normal.clone();
    return clone;
}

void ManyWorlds::SetMin(std::optional<glm::vec3> min)
{
    if (min)
    {
        data->min = *min;
    }
    else
    {
        data->min = glm::vec3(-0.1f);
    }

    data->min_updated = true;
    UpdateShape();
}

void ManyWorlds::SetMax(std::optional<glm::vec3> max)
{
    if (max)
    {
        data->max = *max;
    }
    else
    {
        data->max = glm::vec3(0.1f);
    }
    data->max_updated = true;
    UpdateShape();
}

void ManyWorlds::SetResolution(std::optional<float> resolution)
{
    if (resolution)
    {
        if (*resolution <= 0)
        {
            throw std::invalid_argument("Resolution must be a positive number");
        }

        data->resolution = *resolution;
    }
    else
    {
        data->resolution = 0.001f;
    }
    UpdateShape();
}

void ManyWorlds::SetNSamples(std::optional<int> n_samples)
{
    if (n_samples)
    {
        if (*n_samples <= 0)
        {
            throw std::invalid_argument("Number of samples must be a positive integer");
        }
        data->n_samples = *n_samples;
    }
    else
    {
        data->n_samples = 1;
    }
}

glm::vec3 ManyWorlds::GetMin() const
{
    return data->min;
}

glm::vec3 ManyWorlds::GetMax() const
{
    return data->max;
}

float ManyWorlds::GetResolution() const
{
    return data->resolution;
}

int ManyWorlds::GetNSamples() const
{
    return data->n_samples;
}

torch::Tensor ManyWorlds::GetOccupancy() const
{
    if (data->occupancy.numel() == 0)
    {
        throw std::runtime_error("Occupancy tensor is empty. Please set extent and resolution first.");
    }
    return data->occupancy;
}

torch::Tensor ManyWorlds::GetNormal() const
{
    if (data->normal.numel() == 0)
    {
        throw std::runtime_error("Normal tensor is empty. Please set extent and resolution first.");
    }
    return data->normal;
}

void ManyWorlds::UpdateNormal()
{
    torch::Tensor &normal = data->normal;
    torch::Tensor &occupancy = data->occupancy;

    // Compute gradients along each axis
    // X-gradient
    normal.index({Slice(1, data->shape.x-1), Slice(), Slice(), 0}) =
        0.5f * (occupancy.index({Slice(2, data->shape.x), Slice(), Slice()}) - occupancy.index({Slice(0, data->shape.x-2), Slice(), Slice()}));
    normal.index({0, Slice(), Slice(), 0}) =
        occupancy.index({1, Slice(), Slice()}) - occupancy.index({0, Slice(), Slice()});
    normal.index({data->shape.x-1, Slice(), Slice(), 0}) =
        occupancy.index({data->shape.x-1, Slice(), Slice()}) - occupancy.index({data->shape.x-2, Slice(), Slice()});

    // Y-gradient
    normal.index({Slice(), Slice(1, data->shape.y-1), Slice(), 1}) =
        0.5f * (occupancy.index({Slice(), Slice(2, data->shape.y), Slice()}) - occupancy.index({Slice(), Slice(0, data->shape.y-2), Slice()}));
    normal.index({Slice(), 0, Slice(), 1}) =
        occupancy.index({Slice(), 1, Slice()}) - occupancy.index({Slice(), 0, Slice()});
    normal.index({Slice(), data->shape.y-1, Slice(), 1}) =
        occupancy.index({Slice(), data->shape.y-1, Slice()}) - occupancy.index({Slice(), data->shape.y-2, Slice()});

    // Z-gradient
    normal.index({Slice(), Slice(), Slice(1, data->shape.z-1), 2}) =
        0.5f * (occupancy.index({Slice(), Slice(), Slice(2, data->shape.z)}) - occupancy.index({Slice(), Slice(), Slice(0, data->shape.z-2)}));
    normal.index({Slice(), Slice(), 0, 2}) =
        occupancy.index({Slice(), Slice(), 1}) - occupancy.index({Slice(), Slice(), 0});
    normal.index({Slice(), Slice(), data->shape.z-1, 2}) =
        occupancy.index({Slice(), Slice(), data->shape.z-1}) - occupancy.index({Slice(), Slice(), data->shape.z-2});

    // Normalize the normals
    torch::Tensor norm = normal.norm(2, 3, true);
    torch::Tensor zero_mask = norm == 0;
    torch::Tensor safe_norm = norm.clone();
    safe_norm.masked_fill_(zero_mask, 1.0f);
    normal = normal / safe_norm;
    if (zero_mask.any().item<bool>()) {
        torch::Tensor rand_dirs = torch::randn_like(normal);
        rand_dirs = rand_dirs / rand_dirs.norm(2, 3, true);
        normal.masked_scatter_(zero_mask.expand_as(normal), rand_dirs.masked_select(zero_mask.expand_as(normal)));
    }
}

void ManyWorlds::PrepareForward(Params& params, CUstream stream)
{
    PrepareRendering(params, false, stream);
    params.many_worlds.backward = false;
}


std::pair<torch::Tensor, torch::Tensor> ManyWorlds::PrepareBackward(Params& params, std::optional<torch::Tensor> opt_occupancy_gradient, std::optional<torch::Tensor> opt_normal_gradient, CUstream stream)
{
    PrepareRendering(params, true, stream);
    return AllocateGradTensors(params, opt_occupancy_gradient, opt_normal_gradient);
}


void ManyWorlds::UpdateShape()
{
    glm::ivec3 new_shape = {static_cast<int>(std::ceil((data->max.x - data->min.x) / data->resolution)),
                            static_cast<int>(std::ceil((data->max.y - data->min.y) / data->resolution)),
                            static_cast<int>(std::ceil((data->max.z - data->min.z) / data->resolution))};
    if (new_shape.x <= 0 || new_shape.y <= 0 || new_shape.z <= 0)
    {
        throw std::invalid_argument("Shape dimensions must be positive. Check min, max, and resolution values.");
    }

    if(new_shape.x != data->shape.x || new_shape.y != data->shape.y || new_shape.z != data->shape.z)
    {
        data->shape = new_shape;
        data->max.x = data->min.x + data->shape.x * data->resolution;
        data->max.y = data->min.y + data->shape.y * data->resolution;
        data->max.z = data->min.z + data->shape.z * data->resolution;
        data->occupancy = torch::zeros({data->shape.x, data->shape.y, data->shape.z}, torch::dtype(torch::kFloat).device(torch::kCUDA, 0).requires_grad(true));
        data->normal = torch::zeros({data->shape.x, data->shape.y, data->shape.z, 3}, torch::dtype(torch::kFloat).device(torch::kCUDA, 0));
        UpdateNormal();
    }
}

void ManyWorlds::UpdateBoundingBox(Params& params, CUstream stream)
{
    if(data->min_updated || data->max_updated)
    {
        data->min_updated = false;
        data->max_updated = false;
        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void *>(data->d_mesh), stream));
        torch::Tensor vertices = torch::tensor({
            {data->min.x, data->min.y, data->min.z},
            {data->max.x, data->min.y, data->min.z},
            {data->max.x, data->max.y, data->min.z},
            {data->min.x, data->max.y, data->min.z},
            {data->min.x, data->min.y, data->max.z},
            {data->max.x, data->min.y, data->max.z},
            {data->max.x, data->max.y, data->max.z},
            {data->min.x, data->max.y, data->max.z}
        }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
        torch::Tensor indices = torch::tensor({
            {0, 1, 2}, {0, 2, 3},
            {4, 5, 6}, {4, 6, 7},
            {0, 1, 5}, {0, 5, 4},
            {2, 3, 7}, {2, 7, 6},
            {1, 2, 6}, {1, 6, 5},
            {3, 0, 4}, {3, 4, 7}
        }, torch::dtype(torch::kUInt32).device(torch::kCUDA, 0));
        std::pair<OptixTraversableHandle, CUdeviceptr> gas = OptiX::BuildGAS(vertices, indices, Context::GetInstance().Handle(), stream);
        data->mesh_handle = gas.first;
        data->d_mesh = gas.second;
    }
    params.many_worlds.mesh_handle = data->mesh_handle;
}

void ManyWorlds::UpdateBuffers(Params &params, CUstream stream)
{
    size_t new_buffer_bytes = OPTIX_MAX_GRID_DIM * OPTIX_MAX_GRID_DIM * params.scene.n_receivers * params.scene.signal.n_samples * sizeof(complex3);
    if(new_buffer_bytes != data->buffer_bytes)
    {
        data->buffer_bytes = new_buffer_bytes;
        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void *>(data->d_reference), stream));
        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void *>(data->d_perturbation), stream));
        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void **>(&data->d_reference), data->buffer_bytes, stream));
        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void **>(&data->d_perturbation), data->buffer_bytes, stream));
    }
    params.many_worlds.reference = reinterpret_cast<complex3 *>(data->d_reference);
    params.many_worlds.perturbation = reinterpret_cast<complex3 *>(data->d_perturbation);
}

void ManyWorlds::PrepareRendering(Params& params, bool backward, CUstream stream)
{
    params.many_worlds.min = make_float3(data->min.x, data->min.y, data->min.z);
    params.many_worlds.max = make_float3(data->max.x, data->max.y, data->max.z);
    params.many_worlds.resolution = data->resolution;
    params.many_worlds.n_samples = data->n_samples;
    params.many_worlds.shape = make_int3(data->shape.x, data->shape.y, data->shape.z);
    params.many_worlds.weight = 1.0f / data->n_samples;
    params.many_worlds.backward = backward;
    params.many_worlds.occupancy = reinterpret_cast<float*>(data->occupancy.data_ptr());
    params.many_worlds.normal = reinterpret_cast<float3*>(data->normal.data_ptr());
    if(!(data->occupancy.is_cuda() && data->normal.is_cuda()))
    {
        throw std::runtime_error("Both occupancy and normal tensors must be on the same CUDA device.");
    }
    UpdateBoundingBox(params, stream);
    UpdateBuffers(params, stream);
}

std::pair<torch::Tensor, torch::Tensor> ManyWorlds::AllocateGradTensors(Params &params, std::optional<torch::Tensor> opt_occupancy_gradient, std::optional<torch::Tensor> opt_normal_gradient)
{
    torch::Tensor occupancy_gradient, normal_gradient;
    int3 shape = params.many_worlds.shape;
    if(opt_occupancy_gradient.has_value())
    {
        occupancy_gradient = opt_occupancy_gradient.value();
        if(occupancy_gradient.device().type() != torch::kCUDA)
        {
            throw std::runtime_error("Occupancy gradient tensor must be on CUDA device");
        }
        if(occupancy_gradient.dtype() != torch::kFloat32)
        {
            throw std::runtime_error("Occupancy gradient tensor must have dtype torch::kFloat32");
        }
        if(occupancy_gradient.dim() != 3 || occupancy_gradient.size(0) != shape.x || occupancy_gradient.size(1) != shape.y || occupancy_gradient.size(2) != shape.z)
        {
            throw std::runtime_error("Occupancy gradient tensor must have shape [" + std::to_string(shape.x) + ", " + std::to_string(shape.y) + ", " + std::to_string(shape.z) + "]");
        }
    }
    else
    {
        occupancy_gradient = torch::zeros({shape.x, shape.y, shape.z}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
    }

    if(opt_normal_gradient.has_value())
    {
        normal_gradient = opt_normal_gradient.value();
        if(normal_gradient.device().type() != torch::kCUDA)
        {
            throw std::runtime_error("Normal gradient tensor must be on CUDA device");
        }
        if(normal_gradient.dtype() != torch::kFloat32)
        {
            throw std::runtime_error("Normal gradient tensor must have dtype torch::kFloat32");
        }
        if(normal_gradient.dim() != 4 || normal_gradient.size(0) != shape.x || normal_gradient.size(1) != shape.y || normal_gradient.size(2) != shape.z || normal_gradient.size(3) != 3)
        {
            throw std::runtime_error("Normal gradient tensor must have shape [" + std::to_string(shape.x) + ", " + std::to_string(shape.y) + ", " + std::to_string(shape.z) + ", 3]");
        }
    }
    else
    {
        normal_gradient = torch::zeros({shape.x, shape.y, shape.z, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
    }
    params.many_worlds.occupancy_gradient = reinterpret_cast<float*>(occupancy_gradient.data_ptr());
    params.many_worlds.normal_gradient = reinterpret_cast<float3*>(normal_gradient.data_ptr());
    return {occupancy_gradient, normal_gradient};
}

}