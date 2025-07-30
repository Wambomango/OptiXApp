#include "mwir/antenna_impl.hpp"

#include "mwir/modules/defines.h"



namespace MWIR
{

    AntennaImpl::AntennaImpl()
        : position(0.0f), euler(0.0f), fov(1.0f, 1.0f), ray_density(1.0f)
    {
        UpdateParameters();
    }

    AntennaImpl::AntennaImpl(glm::vec3 position, glm::vec3 euler, glm::vec2 fov, float ray_density)
        : position(position), euler(euler), fov(fov), ray_density(ray_density)
    {
        UpdateParameters();
    }

    AntennaImpl::~AntennaImpl()
    {
    }

    void AntennaImpl::SetPosition(const glm::vec3& position)
    {
        this->position = position;
    }

    void AntennaImpl::SetOrientation(const glm::vec3& euler)
    {
        this->euler = euler;
    }

    void AntennaImpl::SetFOV(const glm::vec2& fov)
    {
        this->fov = fov;
        UpdateParameters();
    }
    
    void AntennaImpl::SetRayDensity(float ray_density)
    {
        this->ray_density = ray_density;
        UpdateParameters();
    }

    glm::vec3 AntennaImpl::GetPosition() const
    {
        return position;
    }

    glm::vec3 AntennaImpl::GetOrientation() const
    {
        return euler;
    }

    glm::vec2 AntennaImpl::GetFOV() const
    {
        return fov;
    }

    float AntennaImpl::GetRayDensity() const
    {
        return ray_density;
    }

    glm::mat3 AntennaImpl::GetRotationMatrix() const
    {
        float cy = cos(euler.z);
        float sy = sin(euler.z);
        float cp = cos(euler.y);
        float sp = sin(euler.y);
        float cr = cos(euler.x);
        float sr = sin(euler.x);

        return glm::mat3(
            cp * cy, cp * sy, -sp,
            sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, sr * cp,
            cr * sp * cy + sr * sy, cr * sp * sy - sr * cy, cr * cp
        );
    }

    float AntennaImpl::GetSolidAngle() const
    {
        return 2 * fov.x * std::sin(fov.y);
    }

    int AntennaImpl::GetNRays() const
    {
        return n_rays;
    }

    int AntennaImpl::GetNBatches() const
    {
        return n_batches;
    }

    void AntennaImpl::UpdateParameters()
    {
        float n_rays_total = ray_density * GetSolidAngle();
        n_batches = std::ceil(n_rays_total / (OPTIX_MAX_GRID_DIM * OPTIX_MAX_GRID_DIM));
        n_rays = n_batches * (OPTIX_MAX_GRID_DIM * OPTIX_MAX_GRID_DIM);
        ray_density = n_rays / GetSolidAngle();
    }
}
