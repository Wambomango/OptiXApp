#pragma once

#include "context.hpp"

namespace OptiX
{
    class Module
    {
    public:
        Module(Context &context, std::string name, std::string path, OptixModuleCompileOptions module_compile_options = {}, OptixPipelineCompileOptions pipeline_compile_options = {}) : name(name)
        {
            std::string content = Utils::ReadFile(path);
            std::string bytecode = Utils::CudaToOptiXIR(name, content);
            OPTIX_CHECK_LOG(optixModuleCreate(
                context.Handle(),
                &module_compile_options,
                &pipeline_compile_options,
                bytecode.c_str(),
                bytecode.size(),
                LOG, &LOG_SIZE,
                &module));
        }

        ~Module()
        {
            OPTIX_CHECK(optixModuleDestroy(module));
        }

        OptixModule Handle()
        {
            return module;
        }

    private:
        std::string name;
        OptixModule module;
    };
}