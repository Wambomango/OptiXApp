#include "optix_renderer.hpp"

#include "utils/optix/utils.hpp"

#include <optix_function_table_definition.h>

OptiXRenderer::OptiXRenderer(Window &window, Scene& scene) 
{
    width = window.GetWidth();
    height = window.GetHeight();
    output_texture = window.GetTexture();

    SetupOptiX();
    SetupGLInterop();
    BuildGAS(scene);
}

OptiXRenderer::~OptiXRenderer() 
{
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(gas_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(params)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(image)));
    CUDA_CHECK(cudaGraphicsUnregisterResource(gl_cuda_resource));
}    


void OptiXRenderer::Render(Camera &camera) 
{
    float azimuth = -glm::radians(camera.GetOrientation().x);
    float elevation = -glm::radians(camera.GetOrientation().y);
    float fov = glm::radians(camera.GetFOV());
    float aspect_ratio = camera.GetAspectRatio();
    float near_plane = camera.GetNearPlane();
    
    glm::vec3 camera_position = camera.GetPosition();
    glm::vec3 camera_w = -glm::normalize(glm::vec3(std::sin(azimuth) * std::cos(elevation), std::sin(elevation), std::cos(azimuth) * std::cos(elevation))) * near_plane;
    glm::vec3 camera_u = glm::normalize(glm::cross(camera_w, glm::vec3(0.0f, 1.0f, 0.0f))) * near_plane * std::tan(fov * 0.5f);
    glm::vec3 camera_v = glm::normalize(glm::cross(camera_u, camera_w)) * near_plane * std::tan(fov * 0.5f) * aspect_ratio;

    Params p;
    p.image = reinterpret_cast<uchar4 *>(image);
    p.image_width = width;
    p.image_height = height;
    p.camera_position = float3{camera_position.x, camera_position.y, camera_position.z};
    p.camera_u = float3{camera_u.x, camera_u.y, camera_u.z};
    p.camera_v = float3{camera_v.x, camera_v.y, camera_v.z};
    p.camera_w = float3{camera_w.x, camera_w.y, camera_w.z};
    p.handle = gas_handle;
    p.light_color = float3{0.8f, 0.8f, 0.8f};
    p.light_direction = float3{0.0f, -1.0f, 0.0f};

    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(params), &p, sizeof(Params), cudaMemcpyHostToDevice, stream));
    OPTIX_CHECK(optixLaunch(pipeline->Handle(), stream, params, sizeof(Params), &sbt, width, height, 1));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaGraphicsMapResources(1, &gl_cuda_resource, 0);
    cudaMemcpy2DToArray(gl_cuda_array, 0, 0, reinterpret_cast<void *>(image), width * sizeof(uchar4), width * sizeof(uchar4), height, cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &gl_cuda_resource, 0);
}


void OptiXRenderer::SetupOptiX() 
{

    OptixModuleCompileOptions module_compile_options = {};
    OptixPipelineCompileOptions pipeline_compile_options = {.usesMotionBlur = false,
                                                            .traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,
                                                            .numPayloadValues = 3,
                                                            .numAttributeValues = 2,
                                                            .exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE,
                                                            .pipelineLaunchParamsVariableName = "params",
                                                            .usesPrimitiveTypeFlags = (unsigned int)OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE};

    module = std::make_unique<OptiX::Module>(ctx, std::string("RenderModule"), MODULE_DIR + std::string("render_module.cu"), module_compile_options, pipeline_compile_options);



    OptixProgramGroupDesc raygen_prog_group_desc = {    .kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
                                                        .raygen = { .module = module->Handle(),
                                                                    .entryFunctionName = "__raygen__rg"}};
    raygen_prog_group = std::make_unique<OptiX::ProgramGroup>(ctx, raygen_prog_group_desc);

    OptixProgramGroupDesc miss_prog_group_desc = {.kind = OPTIX_PROGRAM_GROUP_KIND_MISS,
                                                  .miss = { .module = module->Handle(),
                                                            .entryFunctionName = "__miss__ms"}};
    miss_prog_group = std::make_unique<OptiX::ProgramGroup>(ctx, miss_prog_group_desc);

    OptixProgramGroupDesc hit_prog_group_desc = {  .kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
                                                    .hitgroup = {   .moduleCH = module->Handle(),
                                                                    .entryFunctionNameCH = "__closesthit__ch"}};
    hit_prog_group = std::make_unique<OptiX::ProgramGroup>(ctx, hit_prog_group_desc); 


    std::vector<OptixProgramGroup> program_groups = {raygen_prog_group->Handle(), miss_prog_group->Handle(), hit_prog_group->Handle()};
    pipeline = std::make_unique<OptiX::Pipeline>(ctx, program_groups, pipeline_compile_options, OptixPipelineLinkOptions{.maxTraceDepth = 1});                                            

    raygen_record = std::make_unique<OptiX::SBTRecord<RayGenData>>(ctx, *raygen_prog_group, RayGenData{});
    miss_record = std::make_unique<OptiX::SBTRecord<MissData>>(ctx, *miss_prog_group, MissData{});
    hit_record = std::make_unique<OptiX::SBTRecord<HitData>>(ctx, *hit_prog_group, HitData{});

    sbt = {};
    sbt.raygenRecord = raygen_record->Handle();
    sbt.missRecordBase = miss_record->Handle();
    sbt.missRecordStrideInBytes = miss_record->Size();
    sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = hit_record->Handle();
    sbt.hitgroupRecordStrideInBytes = hit_record->Size();
    sbt.hitgroupRecordCount = 1;                             

    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&params), sizeof(Params)));
}

void OptiXRenderer::SetupGLInterop() 
{
    CUDA_CHECK(cudaGraphicsGLRegisterImage(&gl_cuda_resource, output_texture, GL_TEXTURE_2D,  cudaGraphicsRegisterFlagsSurfaceLoadStore));
    CUDA_CHECK(cudaGraphicsMapResources(1, &gl_cuda_resource, 0));
    CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&gl_cuda_array, gl_cuda_resource, 0, 0));
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &gl_cuda_resource, 0));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&image), width * height * sizeof(uchar4)));
    CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(image), 255, width * height * sizeof(uchar4)));
}

void OptiXRenderer::BuildGAS(Scene &scene) 
{
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;


    auto& attrib = scene.GetAttrib();
    auto& shapes = scene.GetShapes();
    auto& materials = scene.GetMaterials();

    std::vector<float> vertices;
    for (size_t s = 0; s < shapes.size(); s++)
    {
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) 
        {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

            for (size_t v = 0; v < fv; v++) 
            {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                tinyobj::real_t vx = attrib.vertices[3*size_t(idx.vertex_index)+0];
                tinyobj::real_t vy = attrib.vertices[3*size_t(idx.vertex_index)+1];
                tinyobj::real_t vz = attrib.vertices[3*size_t(idx.vertex_index)+2];
                vertices.push_back(vx);
                vertices.push_back(vy);
                vertices.push_back(vz);
            }
            index_offset += fv;
        }
    }


    const size_t vertices_size = sizeof(float) * vertices.size();
    CUdeviceptr d_vertices = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_vertices), vertices_size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_vertices), &vertices[0], vertices_size, cudaMemcpyHostToDevice));

    const uint32_t triangle_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
    OptixBuildInput triangle_input = {};
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.numVertices = static_cast<uint32_t>(vertices.size() / 3);
    triangle_input.triangleArray.vertexBuffers = &d_vertices;
    triangle_input.triangleArray.flags = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords = 1;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(ctx.Handle(), &accel_options, &triangle_input, 1,  &gas_buffer_sizes));

    CUdeviceptr d_temp_buffer_gas;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&gas_buffer), gas_buffer_sizes.outputSizeInBytes));


    OPTIX_CHECK(optixAccelBuild(ctx.Handle(), 0, &accel_options, &triangle_input, 1, d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes, 
                                gas_buffer, gas_buffer_sizes.outputSizeInBytes, &gas_handle, nullptr, 0));

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp_buffer_gas)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_vertices)));
}