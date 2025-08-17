#include "mwir/scene.hpp"

#include "utils/optix/utils.hpp"
#include "utils/optix/torch.hpp"

#include <spdlog/spdlog.h>

namespace MWIR
{

Scene::Scene(std::optional<Mesh> mesh, std::optional<std::vector<Antenna>> senders, std::optional<std::vector<Antenna>> receivers, std::optional<Signal> signal)
{
    data = std::make_shared<SceneData>();
    SetMesh(mesh);
    SetSenders(senders);
    SetReceivers(receivers);
    SetSignal(signal);
}

Scene Scene::Clone() const
{
    std::vector<Antenna> cloned_senders;
    for (const auto& sender : data->senders)
    {
        cloned_senders.push_back(sender.Clone());
    }
    std::vector<Antenna> cloned_receivers;
    for (const auto& receiver : data->receivers)
    {
        cloned_receivers.push_back(receiver.Clone());
    }
    Scene cloned_scene(data->mesh.Clone(), cloned_senders, cloned_receivers, data->signal.Clone());
    return cloned_scene;
}

void Scene::SetMesh(std::optional<Mesh> mesh)
{
    if(mesh.has_value())
    {
        data->mesh = mesh.value();
    }
    else
    {
        data->mesh = Mesh(std::nullopt, std::nullopt);
    }

    data->mesh_updated = true;
}

void Scene::SetSenders(std::optional<std::vector<Antenna>> senders)
{
    if(senders.has_value())
    {
        data->senders = std::move(senders.value());
    }
    else
    {
        data->senders = std::vector<Antenna>();
    }

    data->senders_updated = true;
}

void Scene::SetReceivers(std::optional<std::vector<Antenna>> receivers)
{
    if(receivers.has_value())
    {
        data->receivers = std::move(receivers.value());
    }
    else
    {
        data->receivers = std::vector<Antenna>();
    }

    data->receivers_updated = true;
}

void Scene::SetSignal(std::optional<Signal> signal)
{
    if(signal.has_value())
    {
        data->signal = std::move(signal.value());
    }
    else
    {
        data->signal = Signal(std::nullopt, std::nullopt);
    }

    data->signal_updated = true;
}

Mesh &Scene::GetMesh()
{
    return data->mesh;
}

std::vector<Antenna> &Scene::GetSenders()
{
    return data->senders;
}
std::vector<Antenna> &Scene::GetReceivers()
{
    return data->receivers;
}

Signal &Scene::GetSignal()
{
    return data->signal;
}

void Scene::PrepareRendering(Params &params, CUstream stream)
{
    UpdateMesh(params, stream);
    UpdateSenders(params, stream);
    UpdateReceivers(params, stream);
    UpdateSignal(params, stream);
    UpdateBuffers(params, stream);
}

void Scene::UpdateMesh(Params &params, CUstream stream)
{
    if (data->mesh_updated || data->mesh.data->vertices_updated || data->mesh.data->indices_updated)
    {
        data->mesh_updated = false;
        data->mesh.data->vertices_updated = false;
        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void *>(data->d_mesh), stream));
        torch::Tensor vertices = data->mesh.GetVertices();
        torch::Tensor indices = data->mesh.GetIndices();
        std::pair<OptixTraversableHandle, CUdeviceptr> gas = OptiX::BuildGAS(vertices, indices, Context::GetInstance().Handle(), stream);
        data->mesh_handle = gas.first;
        data->d_mesh = gas.second;
    }
    params.scene.mesh_handle = data->mesh_handle;
}

void Scene::UpdateSenders(Params &params, CUstream stream)
{
    if (data->senders_updated)
    {
        data->senders_updated = false;
        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void *>(data->d_senders), stream));
        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void **>(&data->d_senders), data->senders.size() * sizeof(AntennaData), stream));
        data->h_senders.resize(data->senders.size());
        for (size_t i = 0; i < data->senders.size(); ++i)
        {
            data->h_senders[i].position = float3{data->senders[i].GetPosition().x, data->senders[i].GetPosition().y, data->senders[i].GetPosition().z};
            data->h_senders[i].forward = float3{data->senders[i].GetRotationMatrix()[0].x, data->senders[i].GetRotationMatrix()[0].y, data->senders[i].GetRotationMatrix()[0].z};
            data->h_senders[i].left = float3{data->senders[i].GetRotationMatrix()[1].x, data->senders[i].GetRotationMatrix()[1].y, data->senders[i].GetRotationMatrix()[1].z};
            data->h_senders[i].up = float3{data->senders[i].GetRotationMatrix()[2].x, data->senders[i].GetRotationMatrix()[2].y, data->senders[i].GetRotationMatrix()[2].z};
            data->h_senders[i].fov = float2{data->senders[i].GetFOV().x, data->senders[i].GetFOV().y};
            data->h_senders[i].ray_density = data->senders[i].GetRayDensity();
            data->h_senders[i].solid_angle = data->senders[i].GetSolidAngle();
            data->h_senders[i].n_rays = data->senders[i].GetNRays();
            data->h_senders[i].n_batches = data->senders[i].GetNBatches();
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(data->d_senders), data->h_senders.data(), data->senders.size() * sizeof(AntennaData), cudaMemcpyHostToDevice, stream));
    params.scene.n_senders = static_cast<unsigned int>(data->senders.size());
    params.scene.d_senders = reinterpret_cast<AntennaData *>(data->d_senders);
    params.scene.h_senders = data->h_senders.data();

}

void Scene::UpdateReceivers(Params &params, CUstream stream)
{
    if (data->receivers_updated)
    {
        data->receivers_updated = false;
        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void *>(data->d_receivers), stream));
        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void **>(&data->d_receivers), data->receivers.size() * sizeof(AntennaData), stream));
        data->h_receivers.resize(data->receivers.size());
        for (size_t i = 0; i < data->receivers.size(); ++i)
        {
            data->h_receivers[i].position = float3{data->receivers[i].GetPosition().x, data->receivers[i].GetPosition().y, data->receivers[i].GetPosition().z};
            data->h_receivers[i].forward = float3{data->receivers[i].GetRotationMatrix()[0].x, data->receivers[i].GetRotationMatrix()[0].y, data->receivers[i].GetRotationMatrix()[0].z};
            data->h_receivers[i].left = float3{data->receivers[i].GetRotationMatrix()[1].x, data->receivers[i].GetRotationMatrix()[1].y, data->receivers[i].GetRotationMatrix()[1].z};
            data->h_receivers[i].up = float3{data->receivers[i].GetRotationMatrix()[2].x, data->receivers[i].GetRotationMatrix()[2].y, data->receivers[i].GetRotationMatrix()[2].z};
            data->h_receivers[i].fov = float2{data->receivers[i].GetFOV().x, data->receivers[i].GetFOV().y};
            data->h_receivers[i].ray_density = data->receivers[i].GetRayDensity();
            data->h_receivers[i].solid_angle = data->receivers[i].GetSolidAngle();
            data->h_receivers[i].n_rays = data->receivers[i].GetNRays();
            data->h_receivers[i].n_batches = data->receivers[i].GetNBatches();
        }
        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(data->d_receivers), data->h_receivers.data(), data->receivers.size() * sizeof(AntennaData), cudaMemcpyHostToDevice, stream));
    }
    params.scene.n_receivers = static_cast<unsigned int>(data->receivers.size());
    params.scene.h_receivers = data->h_receivers.data();
    params.scene.d_receivers = reinterpret_cast<AntennaData *>(data->d_receivers);
}

void Scene::UpdateSignal(Params &params, CUstream stream)
{
    if (data->signal_updated)
    {
        data->signal_updated = false;
    }

    params.scene.signal.frequency_range = float2{data->signal.GetFrequencyRange().x, data->signal.GetFrequencyRange().y};
    params.scene.signal.n_samples = data->signal.GetNSamples();
    params.scene.signal.f_step = data->signal.GetFStep();
}

void Scene::UpdateBuffers(Params &params, CUstream stream)
{
    size_t new_buffer_bytes = OPTIX_MAX_GRID_DIM * OPTIX_MAX_GRID_DIM * params.scene.n_receivers * params.scene.signal.n_samples * sizeof(complex3);
    if(new_buffer_bytes != data->buffer_bytes)
    {
        data->buffer_bytes = new_buffer_bytes;
        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void *>(data->d_result), stream));
        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void **>(&data->d_result), data->buffer_bytes, stream));
    }
    CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void *>(data->d_result), 0, data->buffer_bytes, stream));
    params.scene.result = reinterpret_cast<complex3 *>(data->d_result);
}

}