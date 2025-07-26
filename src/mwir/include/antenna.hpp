#pragma once

#include <glm/glm.hpp>

namespace MWIR
{

class AntennaImpl;

class Antenna
{

public:
    Antenna(glm::vec3 position, glm::vec3 euler, glm::vec2 fov, float ray_density);
    ~Antenna();    
    Antenna(const Antenna&) = delete;
    Antenna& operator=(const Antenna&) = delete;
    Antenna(Antenna&&) noexcept;
    Antenna& operator=(Antenna&&) noexcept;

    void SetPosition(const glm::vec3& position);
    void SetOrientation(const glm::vec3& euler);
    void SetFOV(const glm::vec2& fov);
    void SetRayDensity(float ray_density);

    glm::vec3 GetPosition() const;
    glm::vec3 GetOrientation() const;
    glm::vec2 GetFOV() const;
    float GetRayDensity() const;
    glm::mat3 GetRotationMatrix() const;
    float GetSolidAngle() const;
    int GetNRays() const;

protected:
    friend class Scene;
    AntennaImpl *impl; 
};

}