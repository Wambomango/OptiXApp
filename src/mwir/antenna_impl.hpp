#pragma once

#include <glm/glm.hpp>

namespace MWIR
{

class AntennaImpl
{
    public:
        AntennaImpl(glm::vec3 position, glm::vec3 euler, glm::vec2 fov, float ray_density);
        ~AntennaImpl();
        AntennaImpl(const AntennaImpl&) = delete;
        AntennaImpl& operator=(const AntennaImpl&) = delete;
        AntennaImpl(AntennaImpl&&) = default;
        AntennaImpl& operator=(AntennaImpl&&) = default;

        void SetPosition(const glm::vec3& position);
        void SetOrientation(const glm::vec3& orientation);
        void SetFOV(const glm::vec2& fov);
        void SetRayDensity(float ray_density);

        glm::vec3 GetPosition() const;
        glm::vec3 GetOrientation() const;
        glm::vec2 GetFOV() const;
        float GetRayDensity() const;
        glm::mat3 GetRotationMatrix() const;
        float GetSolidAngle() const;
        glm::ivec2 GetNRays() const;

    private:
        void UpdateParameters();

        glm::vec3 position;
        glm::vec3 euler;
        glm::vec2 fov;
        float ray_density;
        glm::vec2 n_rays;
};


}