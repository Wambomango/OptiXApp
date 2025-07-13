#include "mwir/antenna_impl.hpp"

namespace MWIR
{
    AntennaImpl::AntennaImpl(glm::vec3 position, glm::mat3 orientation, glm::vec2 fov)
        : position(position), orientation(orientation), fov(fov)
    {
    }

    AntennaImpl::~AntennaImpl()
    {
    }

    void AntennaImpl::SetPosition(const glm::vec3& position)
    {
        this->position = position;
    }

    void AntennaImpl::SetOrientation(const glm::mat3& orientation)
    {
        this->orientation = orientation;
    }

    void AntennaImpl::SetFOV(const glm::vec2& fov)
    {
        this->fov = fov;
    }

    glm::vec3 AntennaImpl::GetPosition() const
    {
        return position;
    }

    glm::mat3 AntennaImpl::GetOrientation() const
    {
        return orientation;
    }

    glm::vec2 AntennaImpl::GetFOV() const
    {
        return fov;
    }
}