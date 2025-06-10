#pragma once

#include "scene.hpp"
#include "window.hpp"
#include "camera.hpp"

#include "utils/optix/context.hpp"
#include "utils/optix/pipeline.hpp"
#include "utils/optix/module.hpp"
#include "utils/optix/program_group.hpp"
#include "utils/optix/sbt_record.hpp"

#include <optix_types.h>
#include <glad/glad.h>
#include <cuda_gl_interop.h>

#include "render_module.h"


class OptiXRenderer 
{
    public:
        OptiXRenderer(Window &window, Scene& scene);
        ~OptiXRenderer();

        void Render(Camera &camera);

    private:
        void SetupOptiX();
        void SetupGLInterop();
        void BuildGAS(Scene &scene);

        int width;
        int height;

        GLuint output_texture;
        cudaGraphicsResource* gl_cuda_resource;
        cudaArray* gl_cuda_array;

        CUdeviceptr image;
        OptiX::Context ctx;
        std::unique_ptr<OptiX::Module> module;
        std::unique_ptr<OptiX::ProgramGroup> raygen_prog_group;
        std::unique_ptr<OptiX::ProgramGroup> miss_prog_group;
        std::unique_ptr<OptiX::ProgramGroup> hit_prog_group;
        std::unique_ptr<OptiX::Pipeline> pipeline;
        std::unique_ptr<OptiX::SBTRecord<RayGenData>> raygen_record;
        std::unique_ptr<OptiX::SBTRecord<MissData>> miss_record;
        std::unique_ptr<OptiX::SBTRecord<HitData>> hit_record;  
        OptixShaderBindingTable sbt;
        OptixTraversableHandle gas_handle;
        CUdeviceptr gas_buffer;

        CUdeviceptr params;
        CUstream stream;
};