#pragma once

#include "mwir/mesh.hpp"
#include "mwir/antenna.hpp"
#include "mwir/signal.hpp"
#include "mwir/context.hpp"
#include "mwir/modules/render_module.h"

#include <optional>
#include <memory>
#include <vector>
#include <optix_types.h>
#include <cuda_runtime.h>

namespace MWIR
{  

class Scene
{
public:
    Scene(std::optional<Mesh> mesh, std::optional<std::vector<Antenna>> senders, std::optional<std::vector<Antenna>> receivers, std::optional<Signal> signal);
    Scene Clone() const;

    void SetMesh(std::optional<Mesh> mesh);
    void SetSenders(std::optional<std::vector<Antenna>> senders);
    void SetReceivers(std::optional<std::vector<Antenna>> receivers);
    void SetSignal(std::optional<Signal> signal);

    Mesh &GetMesh();
    std::vector<Antenna> &GetSenders();
    std::vector<Antenna> &GetReceivers();
    Signal &GetSignal();

protected:
    friend class ForwardRenderer;
    friend class InverseRenderer;
    void PrepareRendering(Params &params, CUstream stream);

private:
    struct SceneData
    {
        Mesh mesh;
        std::vector<Antenna> senders;
        std::vector<Antenna> receivers;
        Signal signal;

        bool mesh_updated = true;
        bool senders_updated = true;
        bool receivers_updated = true;
        bool signal_updated = true;

        OptixTraversableHandle mesh_handle;
        CUdeviceptr d_mesh = 0;
        std::vector<AntennaData> h_senders;
        CUdeviceptr d_senders = 0;
        std::vector<AntennaData> h_receivers;
        CUdeviceptr d_receivers = 0;
        size_t buffer_bytes = 0;
        CUdeviceptr d_result = 0;

        SceneData() : mesh(std::nullopt, std::nullopt), signal(std::nullopt, std::nullopt)
        {
        }
        ~SceneData()
        {
            CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_mesh)));
            CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_senders)));
            CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_receivers)));
            CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_result)));
        }
    };

    void UpdateMesh(Params &params, CUstream stream);
    void UpdateSenders(Params &params, CUstream stream);
    void UpdateReceivers(Params &params, CUstream stream);
    void UpdateSignal(Params &params, CUstream stream);
    void UpdateBuffers(Params &params, CUstream stream);

    std::shared_ptr<SceneData> data;
};



}