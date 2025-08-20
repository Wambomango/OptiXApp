#pragma once

#include "utils/optix/module.hpp"
#include "utils/optix/program_group.hpp"
#include "utils/optix/pipeline.hpp"
#include "utils/optix/sbt_record.hpp"

#include "mwir/modules/common.h"
#include "mwir/context.hpp"

namespace MWIR
{

class Pipeline
{

public:
    Pipeline();

protected:
    friend class Renderer;

    std::unique_ptr<OptiX::Module> module;
    std::unique_ptr<OptiX::ProgramGroup> raygen_pg;
    std::unique_ptr<OptiX::ProgramGroup> miss_geometry_pg;
    std::unique_ptr<OptiX::ProgramGroup> hit_geometry_pg;
    std::unique_ptr<OptiX::ProgramGroup> miss_antenna_pg;
    std::unique_ptr<OptiX::ProgramGroup> hit_antenna_pg;
    std::unique_ptr<OptiX::Pipeline> pipeline;
    
    std::unique_ptr<OptiX::SBTRecord<RayGenData>> raygen_record;
    std::unique_ptr<OptiX::SBTRecord<MissData>> miss_record;
    std::unique_ptr<OptiX::SBTRecord<HitData>> hit_record;  
    OptixShaderBindingTable sbt;
};

}