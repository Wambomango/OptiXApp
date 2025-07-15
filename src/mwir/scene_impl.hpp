#pragma once

#include "mwir/mesh_impl.hpp"
#include "mwir/antenna_impl.hpp"
#include "mwir/signal_impl.hpp"
#include "mwir/context.hpp"
#include "mwir/modules/render_module.h"

#include <vector>
#include <optix_types.h>
#include <cuda_runtime.h>

namespace MWIR
{  

class SceneImpl
{
public:
    SceneImpl(MeshImpl &&mesh, std::vector<AntennaImpl> &&senders, std::vector<AntennaImpl> &&receivers, SignalImpl &&signal);
    ~SceneImpl();
    SceneImpl(const SceneImpl&) = delete;
    SceneImpl& operator=(const SceneImpl&) = delete;
    SceneImpl(SceneImpl&&) = default;
    SceneImpl& operator=(SceneImpl&&) = default;

    void SetMesh(MeshImpl &&mesh);
    void SetSenders(std::vector<AntennaImpl> &&senders);
    void SetReceivers(std::vector<AntennaImpl> &&receivers);
    void SetSignal(SignalImpl &&signal);

protected:
    friend class RendererImpl;
    void UpdateParams(Params &params);

private:
    void UpdateMesh(Params &params);
    void UpdateSenders(Params &params);
    void UpdateReceivers(Params &params);
    void UpdateSignal(Params &params);

    MeshImpl mesh;
    std::vector<AntennaImpl> senders;
    std::vector<AntennaImpl> receivers;
    SignalImpl signal;

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
};



}