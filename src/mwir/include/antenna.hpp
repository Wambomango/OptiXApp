#pragma once

#include <memory>
#include <glm/glm.hpp>

namespace MWIR
{

class AntennaImpl;

class Antenna
{

public:
    Antenna(glm::vec3 position, glm::mat3 orientation, glm::vec2 fov);
    ~Antenna();    
    Antenna(const Antenna&) = delete;
    Antenna& operator=(const Antenna&) = delete;
    Antenna(Antenna&&) = default;
    Antenna& operator=(Antenna&&) = default;

    void SetPosition(const glm::vec3& position);
    void SetOrientation(const glm::mat3& orientation);
    void SetFOV(const glm::vec2& fov);

    glm::vec3 GetPosition() const;
    glm::mat3 GetOrientation() const;
    glm::vec2 GetFOV() const;

protected:
    friend class Scene;
    std::unique_ptr<AntennaImpl> impl;

 
};

}