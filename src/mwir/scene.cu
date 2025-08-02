#include "mwir/scene.hpp"

#include "utils/optix/utils.hpp"

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
            data->mesh = Mesh(std::nullopt);
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

    SceneParams Scene::GetParams()
    {
        UpdateMesh();
        UpdateSenders();
        UpdateReceivers();
        UpdateSignal();
        return data->params;
    }
    
    void Scene::UpdateMesh()
    {
        if (!data->mesh_updated)
        {
            return;
        }
        data->mesh_updated = false;

        OptiX::Context &ctx = Context::GetInstance();
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(data->d_mesh)));

        torch::Tensor vertices = data->mesh.GetVertices();
        int n_vertices = vertices.size(0);
        if(!vertices.is_cuda())
        {
            vertices = vertices.cuda();
        }
        CUdeviceptr d_vertices = CUdeviceptr(vertices.data_ptr());

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        const uint32_t triangle_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
        OptixBuildInput triangle_input = {};
        triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangle_input.triangleArray.numVertices = n_vertices;
        triangle_input.triangleArray.vertexBuffers = (n_vertices == 0) ? nullptr : &d_vertices;
        triangle_input.triangleArray.flags = triangle_input_flags;
        triangle_input.triangleArray.numSbtRecords = 1;

        OptixAccelBufferSizes mesh_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(ctx.Handle(), &accel_options, &triangle_input, 1,  &mesh_buffer_sizes));

        CUdeviceptr d_temp_buffer_gas;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp_buffer_gas), mesh_buffer_sizes.tempSizeInBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&data->d_mesh), mesh_buffer_sizes.outputSizeInBytes));

        OPTIX_CHECK(optixAccelBuild(ctx.Handle(), 0, &accel_options, &triangle_input, 1, d_temp_buffer_gas, mesh_buffer_sizes.tempSizeInBytes, 
                                    data->d_mesh, mesh_buffer_sizes.outputSizeInBytes, &data->mesh_handle, nullptr, 0));

        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp_buffer_gas)));

        data->params.mesh_handle = data->mesh_handle;
    }
    
    void Scene::UpdateSenders()
    {
        if (!data->senders_updated)
        {
            return;
        }
        data->senders_updated = false;

        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(data->d_senders)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&data->d_senders), data->senders.size() * sizeof(AntennaData)));

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

        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(data->d_senders), data->h_senders.data(), data->senders.size() * sizeof(AntennaData), cudaMemcpyHostToDevice));

        data->params.n_senders = static_cast<unsigned int>(data->senders.size());
        data->params.d_senders = reinterpret_cast<AntennaData *>(data->d_senders);
        data->params.h_senders = data->h_senders.data();
    }

    void Scene::UpdateReceivers()
    {
        if (!data->receivers_updated)
        {
            return;
        }
        data->receivers_updated = false;

        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(data->d_receivers)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&data->d_receivers), data->receivers.size() * sizeof(AntennaData)));

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

        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(data->d_receivers), data->h_receivers.data(), data->receivers.size() * sizeof(AntennaData), cudaMemcpyHostToDevice));

        data->params.n_receivers = static_cast<unsigned int>(data->receivers.size());
        data->params.h_receivers = data->h_receivers.data();
        data->params.d_receivers = reinterpret_cast<AntennaData *>(data->d_receivers);
    }

    void Scene::UpdateSignal()
    {
        if (!data->signal_updated)
        {
            return;
        }
        data->signal_updated = false;

        data->params.signal.frequency_range = float2{data->signal.GetFrequencyRange().x, data->signal.GetFrequencyRange().y};
        data->params.signal.n_samples = data->signal.GetNSamples();
        data->params.signal.f_step = data->signal.GetFStep();
    }
}