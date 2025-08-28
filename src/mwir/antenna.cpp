#include "mwir/antenna.hpp"

#include "mwir/modules/defines.h"


namespace MWIR
{

Antenna::Antenna(std::optional<glm::vec3> position, std::optional<glm::vec3> euler, std::optional<glm::vec2> fov, std::optional<float> ray_density)
{
    data = std::make_shared<AntennaData>();
    SetPosition(position);
    SetOrientation(euler);
    SetFOV(fov);
    SetRayDensity(ray_density);        
    UpdateParameters();
}

Antenna Antenna::Clone() const
{
    return Antenna(data->position, data->euler, data->fov, data->ray_density);
}

void Antenna::SetPosition(std::optional<glm::vec3> position)
{
    if (position.has_value())
    {
        data->position = position.value();
    }
    else
    {
        data->position = glm::vec3(0.0f, 0.0f, 0.0f);
    }
}

void Antenna::SetOrientation(std::optional<glm::vec3> orientation)
{
    if (orientation.has_value())
    {
        data->euler = orientation.value();
    }
    else
    {
        data->euler = glm::vec3(0.0f, 0.0f, 0.0f);
    }
}

void Antenna::SetFOV(std::optional<glm::vec2> fov)
{
    if (fov.has_value())
    {
        data->fov = fov.value();
    }
    else
    {
        data->fov = glm::vec2(1.0f, 1.0f);
    }
    UpdateParameters();
}

void Antenna::SetRayDensity(std::optional<float> ray_density)
{
    if (ray_density.has_value())
    {
        data->ray_density = ray_density.value();
    }
    else
    {
        data->ray_density = 1E9;
    }
    UpdateParameters();
}

glm::vec3 Antenna::GetPosition() const
{
    return data->position;
}

glm::vec3 Antenna::GetOrientation() const
{
    return data->euler;
}

glm::vec2 Antenna::GetFOV() const
{
    return data->fov;
}

float Antenna::GetRayDensity() const
{
    return data->ray_density;
}

glm::mat3 Antenna::GetRotationMatrix() const
{
    float cy = cos(data->euler.z);
    float sy = sin(data->euler.z);
    float cp = cos(data->euler.y);
    float sp = sin(data->euler.y);
    float cr = cos(data->euler.x);
    float sr = sin(data->euler.x);

    return glm::mat3(
        cp * cy, cp * sy, -sp,
        sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, sr * cp,
        cr * sp * cy + sr * sy, cr * sp * sy - sr * cy, cr * cp
    );
}

float Antenna::GetSolidAngle() const
{
    return 2 * data->fov.x * std::sin(data->fov.y);
}

long Antenna::GetNRays() const
{
    return data->n_rays;
}

long Antenna::GetNBatches() const
{
    return data->n_batches;
}

void Antenna::UpdateParameters()
{
    float n_rays_total = data->ray_density * GetSolidAngle();
    data->n_batches = std::ceil(n_rays_total / (OPTIX_MAX_GRID_DIM * OPTIX_MAX_GRID_DIM));
    data->n_rays = data->n_batches * (OPTIX_MAX_GRID_DIM * OPTIX_MAX_GRID_DIM);
    data->ray_density = data->n_rays / GetSolidAngle();
}


}
