#pragma once

#include <glm/glm.hpp>

namespace MWIR
{

class AntennaImpl
{
    public:
        AntennaImpl(glm::vec3 position, glm::mat3 orientation, glm::vec2 fov);
        ~AntennaImpl();
        AntennaImpl(const AntennaImpl&) = delete;
        AntennaImpl& operator=(const AntennaImpl&) = delete;
        AntennaImpl(AntennaImpl&&) = default;
        AntennaImpl& operator=(AntennaImpl&&) = default;

        void SetPosition(const glm::vec3& position);
        void SetOrientation(const glm::mat3& orientation);
        void SetFOV(const glm::vec2& fov);

        glm::vec3 GetPosition() const;
        glm::mat3 GetOrientation() const;
        glm::vec2 GetFOV() const;

    private:
        glm::vec3 position;
        glm::mat3 orientation;
        glm::vec2 fov; 
};


}