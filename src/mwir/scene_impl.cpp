#include "mwir/scene_impl.hpp"

#include "utils/optix/utils.hpp"

#include <spdlog/spdlog.h>

namespace MWIR
{
    SceneImpl::SceneImpl()
        : mesh(MeshImpl()), senders(std::vector<AntennaImpl>(1)), receivers(std::vector<AntennaImpl>(1)), signal(SignalImpl())
    {
    }

    SceneImpl::SceneImpl(MeshImpl &&mesh, std::vector<AntennaImpl> &&senders, std::vector<AntennaImpl> &&receivers, SignalImpl &&signal)
        : mesh(std::move(mesh)), senders(std::move(senders)), receivers(std::move(receivers)), signal(std::move(signal))
    {
    }

    SceneImpl::SceneImpl(SceneImpl&& other)
        : mesh(std::move(other.mesh)), senders(std::move(other.senders)), receivers(std::move(other.receivers)), signal(std::move(other.signal)), params(std::move(other.params)),
          mesh_updated(other.mesh_updated), senders_updated(other.senders_updated), receivers_updated(other.receivers_updated), signal_updated(other.signal_updated),
          mesh_handle(other.mesh_handle), d_mesh(other.d_mesh), h_senders(std::move(other.h_senders)), d_senders(other.d_senders),
          h_receivers(std::move(other.h_receivers)), d_receivers(other.d_receivers)
    {
        other.mesh_handle = {};
        other.d_mesh = 0;
        other.d_senders = 0;
        other.d_receivers = 0;
    }

    SceneImpl& SceneImpl::operator=(SceneImpl&& other)
    {
        if (this != &other)
        {
            mesh = std::move(other.mesh);
            senders = std::move(other.senders);
            receivers = std::move(other.receivers);
            signal = std::move(other.signal);
            params = std::move(other.params);
            mesh_updated = other.mesh_updated;
            senders_updated = other.senders_updated;
            receivers_updated = other.receivers_updated;
            signal_updated = other.signal_updated;
            mesh_handle = other.mesh_handle;
            d_mesh = other.d_mesh;
            h_senders = std::move(other.h_senders);
            d_senders = other.d_senders;
            h_receivers = std::move(other.h_receivers);
            d_receivers = other.d_receivers;

            other.d_mesh = 0;
            other.d_senders = 0;
            other.d_receivers = 0;
        }
        return *this;
    }

    SceneImpl::~SceneImpl()
    {
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_mesh)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_senders)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_receivers)));
    }

    void SceneImpl::SetMesh(MeshImpl &&mesh)
    {
        this->mesh = std::move(mesh);
        mesh_updated = true;
    }

    void SceneImpl::SetSenders(std::vector<AntennaImpl> &&senders)
    {
        this->senders = std::move(senders);
        senders_updated = true;
    }

    void SceneImpl::SetReceivers(std::vector<AntennaImpl> &&receivers)
    {
        this->receivers = std::move(receivers);
        receivers_updated = true;
    }

    void SceneImpl::SetSignal(SignalImpl &&signal)
    {
        this->signal = std::move(signal);
        signal_updated = true;
    }

    MeshImpl SceneImpl::GetMesh()
    {
        MeshImpl tmp = std::move(this->mesh);
        this->mesh = MeshImpl();
        mesh_updated = true;
        return tmp;
    }

    std::vector<AntennaImpl> SceneImpl::GetSenders()
    {
        std::vector<AntennaImpl> tmp = std::move(this->senders);
        this->senders = std::vector<AntennaImpl>(1);
        senders_updated = true;
        return tmp;
    }
    std::vector<AntennaImpl> SceneImpl::GetReceivers()
    {
        std::vector<AntennaImpl> tmp = std::move(this->receivers);
        this->receivers = std::vector<AntennaImpl>(1);
        receivers_updated = true;
        return tmp;
    }

    SignalImpl SceneImpl::GetSignal()
    {
        SignalImpl tmp = std::move(this->signal);
        this->signal = SignalImpl();
        signal_updated = true;
        return tmp;
    }

    SceneParams SceneImpl::GetParams()
    {
        UpdateMesh();
        UpdateSenders();
        UpdateReceivers();
        UpdateSignal();
        return params;
    }
    
    void SceneImpl::UpdateMesh()
    {
        if (!mesh_updated)
        {
            return;
        }
        mesh_updated = false;

        OptiX::Context &ctx = Context::GetInstance();
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_mesh)));

        const std::vector<glm::vec3>& vertices = mesh.GetVertices();
        CUdeviceptr d_vertices = 0;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_vertices), vertices.size() * sizeof(glm::vec3)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_vertices), &vertices[0], vertices.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice));

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        const uint32_t triangle_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
        OptixBuildInput triangle_input = {};
        triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangle_input.triangleArray.numVertices = vertices.size();
        triangle_input.triangleArray.vertexBuffers = &d_vertices;
        triangle_input.triangleArray.flags = triangle_input_flags;
        triangle_input.triangleArray.numSbtRecords = 1;

        OptixAccelBufferSizes mesh_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(ctx.Handle(), &accel_options, &triangle_input, 1,  &mesh_buffer_sizes));

        CUdeviceptr d_temp_buffer_gas;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp_buffer_gas), mesh_buffer_sizes.tempSizeInBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_mesh), mesh_buffer_sizes.outputSizeInBytes));

        OPTIX_CHECK(optixAccelBuild(ctx.Handle(), 0, &accel_options, &triangle_input, 1, d_temp_buffer_gas, mesh_buffer_sizes.tempSizeInBytes, 
                                    d_mesh, mesh_buffer_sizes.outputSizeInBytes, &mesh_handle, nullptr, 0));

        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp_buffer_gas)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_vertices)));

        params.mesh_handle = mesh_handle;
    }
    
    void SceneImpl::UpdateSenders()
    {
        if (!senders_updated)
        {
            return;
        }   
        senders_updated = false;

        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_senders)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_senders), senders.size() * sizeof(AntennaData)));

        h_senders.resize(senders.size());

        for (size_t i = 0; i < senders.size(); ++i)
        {
            h_senders[i].position = float3{senders[i].GetPosition().x, senders[i].GetPosition().y, senders[i].GetPosition().z};
            h_senders[i].forward = float3{senders[i].GetRotationMatrix()[0].x, senders[i].GetRotationMatrix()[0].y, senders[i].GetRotationMatrix()[0].z};
            h_senders[i].left = float3{senders[i].GetRotationMatrix()[1].x, senders[i].GetRotationMatrix()[1].y, senders[i].GetRotationMatrix()[1].z};
            h_senders[i].up = float3{senders[i].GetRotationMatrix()[2].x, senders[i].GetRotationMatrix()[2].y, senders[i].GetRotationMatrix()[2].z};
            h_senders[i].fov = float2{senders[i].GetFOV().x, senders[i].GetFOV().y};
            h_senders[i].ray_density = senders[i].GetRayDensity();
            h_senders[i].solid_angle = senders[i].GetSolidAngle();
            h_senders[i].n_rays = senders[i].GetNRays();
            h_senders[i].n_batches = senders[i].GetNBatches();
        }

        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_senders), h_senders.data(), senders.size() * sizeof(AntennaData), cudaMemcpyHostToDevice));

        params.n_senders = static_cast<unsigned int>(senders.size());
        params.d_senders = reinterpret_cast<AntennaData *>(d_senders);
        params.h_senders = h_senders.data();
    }

    void SceneImpl::UpdateReceivers()
    {
        if (!receivers_updated)
        {
            return;
        }
        receivers_updated = false;

        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_receivers)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_receivers), receivers.size() * sizeof(AntennaData)));

        h_receivers.resize(receivers.size());

        for (size_t i = 0; i < receivers.size(); ++i)
        {
            h_receivers[i].position = float3{receivers[i].GetPosition().x, receivers[i].GetPosition().y, receivers[i].GetPosition().z};
            h_receivers[i].forward = float3{receivers[i].GetRotationMatrix()[0].x, receivers[i].GetRotationMatrix()[0].y, receivers[i].GetRotationMatrix()[0].z};
            h_receivers[i].left = float3{receivers[i].GetRotationMatrix()[1].x, receivers[i].GetRotationMatrix()[1].y, receivers[i].GetRotationMatrix()[1].z};
            h_receivers[i].up = float3{receivers[i].GetRotationMatrix()[2].x, receivers[i].GetRotationMatrix()[2].y, receivers[i].GetRotationMatrix()[2].z};
            h_receivers[i].fov = float2{receivers[i].GetFOV().x, receivers[i].GetFOV().y};
            h_receivers[i].ray_density = receivers[i].GetRayDensity();
            h_receivers[i].solid_angle = receivers[i].GetSolidAngle();
            h_receivers[i].n_rays = receivers[i].GetNRays();
            h_receivers[i].n_batches = receivers[i].GetNBatches();
        }

        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_receivers), h_receivers.data(), receivers.size() * sizeof(AntennaData), cudaMemcpyHostToDevice));

        params.n_receivers = static_cast<unsigned int>(receivers.size());
        params.h_receivers = h_receivers.data();
        params.d_receivers = reinterpret_cast<AntennaData *>(d_receivers);
    }

    void SceneImpl::UpdateSignal()
    {
        if (!signal_updated)
        {
            return;
        }
        signal_updated = false;
   
        params.signal.frequency_range = float2{signal.GetFrequencyRange().x, signal.GetFrequencyRange().y};
        params.signal.n_samples = signal.GetNSamples();
        params.signal.f_step = signal.GetFStep();
    }
}