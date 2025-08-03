#include <torch/torch.h>
#include <spdlog/spdlog.h>

#include "utils/optix/context.hpp"


namespace OptiX
{

std::pair<OptixTraversableHandle, CUdeviceptr> BuildGAS(torch::Tensor vertices, torch::Tensor indices, OptixDeviceContext context)
{
    int n_vertices = vertices.size(0);
    int n_triangles = indices.size(0);

    if(!vertices.is_cuda())
    {
        vertices = vertices.cuda();
    }
    if(!indices.is_cuda())
    {
        indices = indices.cuda();
    }

    CUdeviceptr d_vertices = CUdeviceptr(vertices.data_ptr());
    CUdeviceptr d_indices = CUdeviceptr(indices.data_ptr());

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
    triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangle_input.triangleArray.indexStrideInBytes = 12; 
    triangle_input.triangleArray.numIndexTriplets = n_triangles;
    triangle_input.triangleArray.indexBuffer = (n_triangles == 0) ? (CUdeviceptr)nullptr : d_indices;

    OptixAccelBufferSizes mesh_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options, &triangle_input, 1,  &mesh_buffer_sizes));

    CUdeviceptr d_temp_buffer_gas;
    CUdeviceptr d_mesh;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp_buffer_gas), mesh_buffer_sizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_mesh), mesh_buffer_sizes.outputSizeInBytes));

    OptixTraversableHandle mesh_handle;
    OPTIX_CHECK(optixAccelBuild(context, 0, &accel_options, &triangle_input, 1, d_temp_buffer_gas, mesh_buffer_sizes.tempSizeInBytes, d_mesh, mesh_buffer_sizes.outputSizeInBytes, &mesh_handle, nullptr, 0));

    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp_buffer_gas)));

    return {mesh_handle, d_mesh};
}


}


       


