#include "mwir/include/antenna.hpp"
#include "mwir/antenna_impl.hpp"

namespace MWIR
{

Antenna::Antenna(glm::vec3 position, glm::mat3 orientation, glm::vec2 fov) : impl(std::make_unique<AntennaImpl>(position, orientation, fov))
{
}

Antenna::~Antenna()
{
}

void Antenna::SetPosition(const glm::vec3& position)
{
    impl->SetPosition(position);
}

void Antenna::SetOrientation(const glm::mat3& orientation)
{
    impl->SetOrientation(orientation);
}

void Antenna::SetFOV(const glm::vec2& fov)
{
    impl->SetFOV(fov);
}

glm::vec3 Antenna::GetPosition() const
{
    return impl->GetPosition();
}

glm::mat3 Antenna::GetOrientation() const
{
    return impl->GetOrientation();
}

glm::vec2 Antenna::GetFOV() const
{
    return impl->GetFOV();
}

}