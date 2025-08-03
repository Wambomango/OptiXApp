#include "mwir/many_worlds.hpp"
#include <spdlog/spdlog.h>

using torch::indexing::Slice;

namespace MWIR
{

ManyWorlds::ManyWorlds(std::optional<glm::vec3> min, std::optional<glm::vec3> max, std::optional<float> resolution)
{
    data = std::make_shared<ManyWorldsData>();
    data->min = min.value_or(glm::vec3(-0.1f));
    data->max = max.value_or(glm::vec3(0.1f));
    data->resolution = resolution.value_or(0.001f);
    UpdateParams();
}

ManyWorlds ManyWorlds::Clone() const 
{
    ManyWorlds clone(data->min, data->max, data->resolution);
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
    UpdateParams();
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
    UpdateParams();
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
    UpdateParams();
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

void ManyWorlds::UpdateNormals()
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

void ManyWorlds::UpdateParams()
{
    data->shape = {static_cast<int>(std::ceil((data->max.x - data->min.x) / data->resolution)),
            static_cast<int>(std::ceil((data->max.y - data->min.y) / data->resolution)),
            static_cast<int>(std::ceil((data->max.z - data->min.z) / data->resolution))};

    if (data->shape.x <= 0 || data->shape.y <= 0 || data->shape.z <= 0)
    {
        throw std::invalid_argument("Shape dimensions must be positive. Check min, max, and resolution values.");
    }

    data->max.x = data->min.x + data->shape[0] * data->resolution;
    data->max.y = data->min.y + data->shape[1] * data->resolution;
    data->max.z = data->min.z + data->shape[2] * data->resolution;

    if (data->shape[0] != data->occupancy.size(0) ||  data->shape[1] != data->occupancy.size(1) || data->shape[2] != data->occupancy.size(2))
    {
        data->occupancy = torch::zeros({data->shape[0], data->shape[1], data->shape[2]}, torch::dtype(torch::kFloat).device(torch::kCUDA, 0).requires_grad(true));
        data->normal = torch::zeros({data->shape[0], data->shape[1], data->shape[2], 3}, torch::dtype(torch::kFloat).device(torch::kCUDA, 0));
        UpdateNormals();
    }
}


}