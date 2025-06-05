#pragma once

#include "context.hpp"

namespace OptiX
{
    class ProgramGroup
    {
    public:
        ProgramGroup(Context &context, OptixProgramGroupDesc prog_group_desc, OptixProgramGroupOptions program_group_options = {}, OptixPipelineCompileOptions pipeline_compile_options = {})
        {
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                context.Handle(),
                &prog_group_desc,
                1, // num program groups
                &program_group_options,
                LOG, &LOG_SIZE,
                &prog_group));
        }

        ~ProgramGroup()
        {
            OPTIX_CHECK(optixProgramGroupDestroy(prog_group));
        }

        OptixProgramGroup Handle()
        {
            return prog_group;
        }

    private:
        OptixProgramGroup prog_group;
    };
}