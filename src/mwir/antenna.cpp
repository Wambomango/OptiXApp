#include "mwir/include/antenna.hpp"
#include "mwir/antenna_impl.hpp"

namespace MWIR
{

Antenna::Antenna()
{
    impl = new AntennaImpl();
}

Antenna::Antenna(glm::vec3 position, glm::vec3 euler, glm::vec2 fov, float ray_density)
{
    impl = new AntennaImpl(position, euler, fov, ray_density);
}

Antenna::Antenna(AntennaImpl &&antenna_impl)
{
    impl = new AntennaImpl(std::move(antenna_impl));
}

Antenna::~Antenna()
{
    if(impl)
    {
        delete impl;
    }
}

Antenna::Antenna(Antenna&& other) noexcept : impl(other.impl)
{
    other.impl = nullptr; 
}

Antenna& Antenna::operator=(Antenna&& other) noexcept
{
    if (this != &other)
    {
        if(impl)
        {
            delete impl; 
        } 
        impl = other.impl; 
        other.impl = nullptr; 
    }
    return *this;
}

void Antenna::SetPosition(const glm::vec3& position)
{
    if(impl)
    {
        impl->SetPosition(position);
    }
}

void Antenna::SetOrientation(const glm::vec3& euler)
{
    if(impl)
    {
        impl->SetOrientation(euler);
    }
}

void Antenna::SetFOV(const glm::vec2& fov)
{
    if(impl)
    {
        impl->SetFOV(fov);
    }
}

void Antenna::SetRayDensity(float ray_density)
{
    if(impl)
    {
        impl->SetRayDensity(ray_density);
    }
}

glm::vec3 Antenna::GetPosition() const
{
    if(impl)
    {
        return impl->GetPosition();
    }
    return glm::vec3(0.0f);
}

glm::vec3 Antenna::GetOrientation() const
{
    if(impl)
    {
        return impl->GetOrientation();
    }
    return glm::vec3(0.0f);
}

glm::vec2 Antenna::GetFOV() const
{
    if(impl)
    {
        return impl->GetFOV();
    }
    return glm::vec2(0.0f);
}

float Antenna::GetRayDensity() const
{
    if(impl)
    {
        return impl->GetRayDensity();
    }
    return 0.0f;
}

glm::mat3 Antenna::GetRotationMatrix() const
{
    if(impl)
    {
        return impl->GetRotationMatrix();
    }
    return glm::mat3(0.0f);
}

float Antenna::GetSolidAngle() const
{
    if(impl)
    {
        return impl->GetSolidAngle();
    }
    return 0.0f;
}

int Antenna::GetNRays() const
{
    if(impl)
    {
        return impl->GetNRays();
    }
    return 0;
}

}