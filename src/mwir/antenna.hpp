#pragma once

#include <glm/glm.hpp>
#include <spdlog/spdlog.h>
#include <optional>

namespace MWIR
{

class Antenna
{
    public:
        Antenna(std::optional<glm::vec3> position, std::optional<glm::vec3> euler, std::optional<glm::vec2> fov, std::optional<float> ray_density);
        Antenna Clone() const;

        void SetPosition(std::optional<glm::vec3> position);
        void SetOrientation(std::optional<glm::vec3> orientation);
        void SetFOV(std::optional<glm::vec2> fov);
        void SetRayDensity(std::optional<float> ray_density);

        glm::vec3 GetPosition() const;
        glm::vec3 GetOrientation() const;
        glm::vec2 GetFOV() const;
        float GetRayDensity() const;
        glm::mat3 GetRotationMatrix() const;
        float GetSolidAngle() const;
        int GetNRays() const;
        int GetNBatches() const;

    private:
        void UpdateParameters();

        struct AntennaData
        {
            glm::vec3 position;
            glm::vec3 euler;
            glm::vec2 fov;
            float ray_density;
            int n_rays;
            int n_batches;
        };

        std::shared_ptr<AntennaData> data;
};


}