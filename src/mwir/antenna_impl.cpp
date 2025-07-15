#include "mwir/antenna_impl.hpp"

namespace MWIR
{
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

    glm::ivec2 AntennaImpl::GetNRays() const
    {
        return n_rays;
    }

    void AntennaImpl::UpdateParameters()
    {
        int n_rays_total = ray_density * GetSolidAngle();
        int n_x = std::sqrt(n_rays_total * fov.x / fov.y);
        int n_y = std::ceil(float(n_rays_total) / n_x);
        n_rays = glm::ivec2(n_x, n_y);
        ray_density = n_x * n_y / GetSolidAngle();
    }

}
