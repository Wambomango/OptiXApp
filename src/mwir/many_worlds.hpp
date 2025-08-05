#pragma once

#include "mwir/modules/render_module.h"

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
        ManyWorlds(std::optional<glm::vec3> min, std::optional<glm::vec3> max, std::optional<float> resolution, std::optional<int> n_samples);
        ManyWorlds Clone() const;

        void SetMin(std::optional<glm::vec3> min);
        void SetMax(std::optional<glm::vec3> max);
        void SetResolution(std::optional<float> resolution);
        void SetNSamples(std::optional<int> n_samples);
        glm::vec3 GetMin() const;
        glm::vec3 GetMax() const;
        float GetResolution() const;
        torch::Tensor GetOccupancy() const;
        torch::Tensor GetNormal() const;
        int GetNSamples() const;

        void UpdateNormal();
        
    protected:
        friend class InverseRenderer;
        ManyWorldsParams GetParams();
        
    private:
        struct ManyWorldsData
        {
            glm::vec3 min;
            glm::vec3 max;
            float resolution;
            int n_samples;
            glm::ivec3 shape;
            bool min_updated = true;
            bool max_updated = true;
            torch::Tensor occupancy;
            torch::Tensor normal;
            OptixTraversableHandle mesh_handle;
            CUdeviceptr d_mesh = 0;
            ManyWorldsParams params;
        };    
    
        void UpdateParameters();
        void UpdateBBMesh();

        std::shared_ptr<ManyWorldsData> data;
};


}