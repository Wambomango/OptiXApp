#pragma once

#include "context.hpp"

namespace OptiX
{
    class Pipeline
    {
    public:
        Pipeline(Context &context, std::vector<OptixProgramGroup>& program_groups, OptixPipelineCompileOptions pipeline_compile_options = {}, OptixPipelineLinkOptions pipeline_link_options = {})
        {
        OPTIX_CHECK_LOG(optixPipelineCreate(
            context.Handle(),
            &pipeline_compile_options,
            &pipeline_link_options,
            program_groups.data(),
            program_groups.size(),
            LOG, &LOG_SIZE,
            &pipeline));

            OptixStackSizes stack_sizes = {};
            for (auto &prog_group : program_groups)
            {
                OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes, pipeline));
            }

            uint32_t direct_callable_stack_size_from_traversal;
            uint32_t direct_callable_stack_size_from_state;
            uint32_t continuation_stack_size;
            OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, pipeline_link_options.maxTraceDepth,
                                                0, // maxCCDepth
                                                0, // maxDCDEpth
                                                &direct_callable_stack_size_from_traversal,
                                                &direct_callable_stack_size_from_state, &continuation_stack_size));
            OPTIX_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
                                                direct_callable_stack_size_from_state, continuation_stack_size,
                                                2 // maxTraversableDepth
                                                ));
  
        }

        ~Pipeline()
        {
            OPTIX_CHECK(optixPipelineDestroy(pipeline));
        }

        OptixPipeline Handle()
        {
            return pipeline;
        }

    private:
        OptixPipeline pipeline;
    };
}