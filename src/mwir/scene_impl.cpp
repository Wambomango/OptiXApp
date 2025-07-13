#include "mwir/scene_impl.hpp"

#include "utils/optix/utils.hpp"

namespace MWIR
{
    SceneImpl::SceneImpl(MeshImpl &&mesh, std::vector<AntennaImpl> &&senders, std::vector<AntennaImpl> &&receivers, SignalImpl &&signal)
        : mesh(std::move(mesh)), senders(std::move(senders)), receivers(std::move(receivers)), signal(std::move(signal))
    {
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

    void SceneImpl::UpdateParams(Params &params)
    {
        OptiX::Context &ctx = Context::GetInstance();
        CUDA_CHECK(cudaFree(0));

        UpdateMesh(params);
        UpdateSenders(params);
        UpdateReceivers(params);
        UpdateSignal(params);
    }
    
    void SceneImpl::UpdateMesh(Params &params)
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
    
    void SceneImpl::UpdateSenders(Params &params)
    {
        if (!senders_updated)
        {
            return;
        }   
        senders_updated = false;

        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_senders)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_senders), senders.size() * sizeof(AntennaData)));

        AntennaData h_senders[senders.size()];
        for (size_t i = 0; i < senders.size(); ++i)
        {
            h_senders[i].position = float3{senders[i].GetPosition().x, senders[i].GetPosition().y, senders[i].GetPosition().z};
            h_senders[i].forward = float3{senders[i].GetOrientation()[0].x, senders[i].GetOrientation()[0].y, senders[i].GetOrientation()[0].z};
            h_senders[i].left = float3{senders[i].GetOrientation()[1].x, senders[i].GetOrientation()[1].y, senders[i].GetOrientation()[1].z};
            h_senders[i].up = float3{senders[i].GetOrientation()[2].x, senders[i].GetOrientation()[2].y, senders[i].GetOrientation()[2].z};
            h_senders[i].fov = float2{senders[i].GetFOV().x, senders[i].GetFOV().y};
        }

        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_senders), h_senders, senders.size() * sizeof(AntennaData), cudaMemcpyHostToDevice));

        params.n_senders = static_cast<unsigned int>(senders.size());
        params.senders = reinterpret_cast<AntennaData *>(d_senders);
    }

    void SceneImpl::UpdateReceivers(Params &params)
    {
        if (!receivers_updated)
        {
            return;
        }
        receivers_updated = false;

        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_receivers)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_receivers), receivers.size() * sizeof(AntennaData)));

        AntennaData h_receivers[receivers.size()];
        for (size_t i = 0; i < receivers.size(); ++i)
        {
            h_receivers[i].position = float3{receivers[i].GetPosition().x, receivers[i].GetPosition().y, receivers[i].GetPosition().z};
            h_receivers[i].forward = float3{receivers[i].GetOrientation()[0].x, receivers[i].GetOrientation()[0].y, receivers[i].GetOrientation()[0].z};
            h_receivers[i].left = float3{receivers[i].GetOrientation()[1].x, receivers[i].GetOrientation()[1].y, receivers[i].GetOrientation()[1].z};
            h_receivers[i].up = float3{receivers[i].GetOrientation()[2].x, receivers[i].GetOrientation()[2].y, receivers[i].GetOrientation()[2].z};
            h_receivers[i].fov = float2{receivers[i].GetFOV().x, receivers[i].GetFOV().y};
        }

        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_receivers), h_receivers, receivers.size() * sizeof(AntennaData), cudaMemcpyHostToDevice));

        params.n_receivers = static_cast<unsigned int>(receivers.size());
        params.receivers = reinterpret_cast<AntennaData *>(d_receivers);
    }

    void SceneImpl::UpdateSignal(Params &params)
    {
        if (!signal_updated)
        {
            return;
        }
        signal_updated = false;
   
        params.signal.frequency_range = float2{signal.GetFrequencyRange().x, signal.GetFrequencyRange().y};
        params.signal.n_frequencies = signal.GetNFrequencies();
        params.signal.f_step = signal.GetFStep();
    }
}