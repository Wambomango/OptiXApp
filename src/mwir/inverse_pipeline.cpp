#include "mwir/inverse_pipeline.hpp"

namespace MWIR
{

InversePipeline::InversePipeline()
{
    OptiX::Context &ctx = Context::GetInstance();

    OptixModuleCompileOptions module_compile_options = {};
    OptixPipelineCompileOptions pipeline_compile_options = {.usesMotionBlur = false,
                                                            .traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,
                                                            .numPayloadValues = 2,
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
}


}
