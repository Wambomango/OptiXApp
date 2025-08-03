#pragma once

#include <torch/torch.h>
#include <glm/glm.hpp>
#include <string>
#include <vector>
#include <optional>
#include <memory>

namespace MWIR
{

class ManyWorlds
{
    public:
        ManyWorlds(std::optional<glm::vec3> min, std::optional<glm::vec3> max, std::optional<float> resolution);
        ManyWorlds Clone() const;

        void SetMin(std::optional<glm::vec3> min);
        void SetMax(std::optional<glm::vec3> max);
        void SetResolution(std::optional<float> resolution);
        glm::vec3 GetMin() const;
        glm::vec3 GetMax() const;
        float GetResolution() const;
        torch::Tensor GetOccupancy() const;
        torch::Tensor GetNormal() const;

        void UpdateNormals();
        
    private:
        struct ManyWorldsData
        {
            glm::vec3 min;
            glm::vec3 max;
            float resolution;
            glm::ivec3 shape;
            torch::Tensor occupancy;
            torch::Tensor normal;
        };    
    
        void UpdateParams();

        std::shared_ptr<ManyWorldsData> data;
};


}