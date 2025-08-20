#pragma once

#include "mwir/modules/common.h"
#include "mwir/modules/defines.h"

#include "utils/optix/utils.hpp"

#include <torch/torch.h>
#include <glm/glm.hpp>
#include <string>
#include <vector>
#include <optional>
#include <memory>
#include <cuda.h>

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
        friend class ManyWorldsRenderer;
        void PrepareForward(Params& params, CUstream stream);
        std::pair<torch::Tensor, torch::Tensor> PrepareBackward(Params& params, torch::Tensor &e_field_gradient, std::optional<torch::Tensor> opt_occupancy_gradient, std::optional<torch::Tensor> opt_normal_gradient, CUstream stream);

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
            size_t buffer_bytes = 0;
            CUdeviceptr d_reference = 0;
            CUdeviceptr d_perturbation = 0;

            ~ManyWorldsData() 
            {
                CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_mesh)));
                CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_reference)));
                CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_perturbation)));
            }
        };

        void PrepareRendering(Params& params, bool backward, CUstream stream);
        void UpdateShape();
        void UpdateBoundingBox(Params& params, CUstream stream);
        void UpdateBuffers(Params& params, CUstream stream);
        std::pair<torch::Tensor, torch::Tensor> AllocateGradTensors(Params &params, std::optional<torch::Tensor> opt_occupancy_gradient, std::optional<torch::Tensor> opt_normal_gradient);


        std::shared_ptr<ManyWorldsData> data;
};


}