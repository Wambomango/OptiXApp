#include "mwir/many_worlds_pipeline.hpp"

namespace MWIR
{

ManyWorldsPipeline::ManyWorldsPipeline()
{
    OptiX::Context &ctx = Context::GetInstance();

    OptixModuleCompileOptions module_compile_options = {};//{.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, .debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL};
    OptixPipelineCompileOptions pipeline_compile_options = {.usesMotionBlur = false,
                                                            .traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,
                                                            .numPayloadValues = 2,
                                                            .numAttributeValues = 2,
                                                            .exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE,
                                                            .pipelineLaunchParamsVariableName = "params",
                                                            .usesPrimitiveTypeFlags = (unsigned int)OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE};

    module = std::make_unique<OptiX::Module>(ctx, std::string("ManyWorldsModule"), MODULE_DIR + std::string("many_worlds_render_module.cu"), module_compile_options, pipeline_compile_options);

   OptixProgramGroupDesc raygen_pg_desc = {.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
                                            .raygen = {.module = module->Handle(),
                                                       .entryFunctionName = "__raygen__rg"}};
    raygen_pg = std::make_unique<OptiX::ProgramGroup>(ctx, raygen_pg_desc);

    OptixProgramGroupDesc miss_geometry_pg_desc = {.kind = OPTIX_PROGRAM_GROUP_KIND_MISS,
                                                   .miss = {.module = module->Handle(),
                                                            .entryFunctionName = "__miss__geometry"}};
    miss_geometry_pg = std::make_unique<OptiX::ProgramGroup>(ctx, miss_geometry_pg_desc);

    OptixProgramGroupDesc hit_geometry_pg_desc = {.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
                                                  .hitgroup = {.moduleCH = module->Handle(),
                                                               .entryFunctionNameCH = "__closesthit__geometry"}};
    hit_geometry_pg = std::make_unique<OptiX::ProgramGroup>(ctx, hit_geometry_pg_desc);


    OptixProgramGroupDesc miss_antenna_pg_desc = {.kind = OPTIX_PROGRAM_GROUP_KIND_MISS,
                                                  .miss = {.module = module->Handle(),
                                                           .entryFunctionName = "__miss__antenna"}};
    miss_antenna_pg = std::make_unique<OptiX::ProgramGroup>(ctx, miss_antenna_pg_desc);

    OptixProgramGroupDesc hit_antenna_pg_desc = {.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
                                                 .hitgroup = {.moduleCH = module->Handle(),
                                                             .entryFunctionNameCH = "__closesthit__antenna"}};
    hit_antenna_pg = std::make_unique<OptiX::ProgramGroup>(ctx, hit_antenna_pg_desc);



    OptixProgramGroupDesc miss_manyworlds_pg_desc = {.kind = OPTIX_PROGRAM_GROUP_KIND_MISS,
                                                  .miss = {.module = module->Handle(),
                                                           .entryFunctionName = "__miss__manyworlds"}};
    miss_manyworlds_pg = std::make_unique<OptiX::ProgramGroup>(ctx, miss_manyworlds_pg_desc);

    OptixProgramGroupDesc hit_manyworlds_pg_desc = {.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
                                                 .hitgroup = {.moduleCH = module->Handle(),
                                                              .entryFunctionNameCH = "__closesthit__manyworlds"}};
    hit_manyworlds_pg = std::make_unique<OptiX::ProgramGroup>(ctx, hit_manyworlds_pg_desc);

    
    std::vector<OptixProgramGroup> pgs = {raygen_pg->Handle(), 
                                          miss_geometry_pg->Handle(), hit_geometry_pg->Handle(), 
                                          miss_antenna_pg->Handle(), hit_antenna_pg->Handle(), 
                                          miss_manyworlds_pg->Handle(), hit_manyworlds_pg->Handle()};
    pipeline = std::make_unique<OptiX::Pipeline>(ctx, pgs, pipeline_compile_options, OptixPipelineLinkOptions{.maxTraceDepth = 2});

    std::vector<OptixProgramGroup> raygen_pgs = {raygen_pg->Handle()};
    raygen_record = std::make_unique<OptiX::SBTRecord<RayGenData>>(ctx, raygen_pgs, std::vector<RayGenData>{RayGenData{}});
    std::vector<OptixProgramGroup> miss_pgs = {miss_geometry_pg->Handle(), miss_antenna_pg->Handle(), miss_manyworlds_pg->Handle()};
    miss_record = std::make_unique<OptiX::SBTRecord<MissData>>(ctx, miss_pgs, std::vector<MissData>{MissData{}, MissData{}, MissData{}});
    std::vector<OptixProgramGroup> hit_pgs = {hit_geometry_pg->Handle(), hit_antenna_pg->Handle(), hit_manyworlds_pg->Handle()};
    hit_record = std::make_unique<OptiX::SBTRecord<HitData>>(ctx, hit_pgs, std::vector<HitData>{HitData{}, HitData{}, HitData{}});

    sbt = {};
    sbt.raygenRecord = raygen_record->Handle();
    sbt.missRecordBase = miss_record->Handle();
    sbt.missRecordStrideInBytes = miss_record->ElementSize();
    sbt.missRecordCount = miss_record->NumElements();
    sbt.hitgroupRecordBase = hit_record->Handle();
    sbt.hitgroupRecordStrideInBytes = hit_record->ElementSize();
    sbt.hitgroupRecordCount = hit_record->NumElements();                                                
}


}
