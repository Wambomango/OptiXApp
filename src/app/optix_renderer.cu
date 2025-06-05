#include "optix_renderer.hpp"

#include "utils/optix/utils.hpp"

#include <optix_function_table_definition.h>

// #include "modules/optixHello.h"

// template <typename T>
// struct SbtRecord
// {
//     __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
//     T data;
// };

// typedef SbtRecord<RayGenData> RayGenSbtRecord;
// typedef SbtRecord<int> MissSbtRecord;



OptiXRenderer::OptiXRenderer(Window &window, Scene& scene) 
{
    width = window.GetWidth();
    height = window.GetHeight();
    output_texture = window.GetTexture();

    cudaGraphicsGLRegisterImage(&gl_cuda_resource, output_texture, GL_TEXTURE_2D,  cudaGraphicsRegisterFlagsSurfaceLoadStore);
    cudaGraphicsMapResources(1, &gl_cuda_resource, 0);
    cudaGraphicsSubResourceGetMappedArray(&gl_cuda_array, gl_cuda_resource, 0, 0);
    cudaGraphicsUnmapResources(1, &gl_cuda_resource, 0);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&image), width * height * sizeof(uchar4)));
    CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(image), 255, width * height * sizeof(uchar4)));

}

OptiXRenderer::~OptiXRenderer() 
{
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(image)));
    cudaGraphicsUnregisterResource(gl_cuda_resource);
}    


void OptiXRenderer::Render(Camera &camera) 
{
    cudaGraphicsMapResources(1, &gl_cuda_resource, 0);
    cudaMemcpy2DToArray(gl_cuda_array, 0, 0, reinterpret_cast<void *>(image), width * sizeof(uchar4), width * sizeof(uchar4), height, cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &gl_cuda_resource, 0);




    // OptiX::Context ctx;
    // OptixPipelineCompileOptions pipeline_compile_options = {.usesMotionBlur = false,
    //                                                         .traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
    //                                                         .numPayloadValues = 2,
    //                                                         .numAttributeValues = 2,
    //                                                         .exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE,
    //                                                         .pipelineLaunchParamsVariableName = "params",
    //                                                         .usesPrimitiveTypeFlags = (unsigned int)OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE};

    // OptiX::Module module(ctx, "Test", "/home/mario/Desktop/Masterarbeit/OptiXApp/src/modules/draw_solid_color.cu", {}, pipeline_compile_options);

    // OptixProgramGroupDesc raygen_prog_group_desc = {
    //     .kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
    //     .raygen = {
    //         .module = module.Handle(),
    //         .entryFunctionName = "__raygen__draw_solid_color"}};
    // OptiX::ProgramGroup raygen_prog_group(ctx, raygen_prog_group_desc);


    // OptixProgramGroupDesc miss_prog_group_desc = {
    //     .kind = OPTIX_PROGRAM_GROUP_KIND_MISS};
    // OptiX::ProgramGroup miss_prog_group(ctx, miss_prog_group_desc);

    
    // std::vector<OptixProgramGroup> program_groups = {raygen_prog_group.Handle(), miss_prog_group.Handle()};
    // OptiX::Pipeline pipeline(ctx, program_groups, pipeline_compile_options);


    // OptiX::SBTRecord<RayGenSbtRecord> raygen_record(ctx, raygen_prog_group);
    // OptiX::SBTRecord<MissSbtRecord> miss_record(ctx, miss_prog_group);  
    // OptixShaderBindingTable sbt = {};
    // sbt.raygenRecord = raygen_record.Handle();
    // sbt.missRecordBase = miss_record.Handle();
    // sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    // sbt.missRecordCount = 1;

    // CUdeviceptr image;
    // CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&image), width * height * sizeof(uchar4)));
    // CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(image), 255, width * height * sizeof(uchar4)));

    // CUDA_CHECK(cudaFree(reinterpret_cast<void *>(image)));










    // {
    //     CUstream stream;
    //     CUDA_CHECK(cudaStreamCreate(&stream));

    //     Params params;
    //     params.image = reinterpret_cast<uchar4 *>(image);
    //     params.image_width = width;

    //     CUdeviceptr d_param;
    //     CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_param), sizeof(Params)));
    //     CUDA_CHECK(cudaMemcpy(
    //         reinterpret_cast<void *>(d_param),
    //         &params, sizeof(params),
    //         cudaMemcpyHostToDevice));

    //     OPTIX_CHECK(optixLaunch(pipeline.Handle(), stream, d_param, sizeof(Params), &sbt, width, height, /*depth=*/1));
    //     CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_param)));
    // }

    // {
    //     uint8_t *image_host = new uint8_t[width * height * sizeof(uchar4)];
    //     cudaDeviceSynchronize();
    //     CUDA_CHECK(cudaMemcpy(image_host, reinterpret_cast<void **>(image), width * height * sizeof(uchar4), cudaMemcpyDeviceToHost));

    //     SPDLOG_WARN("{} {} {} {}", image_host[0], image_host[1], image_host[2], image_host[3]);

    //     delete[] image_host;
    // }
}