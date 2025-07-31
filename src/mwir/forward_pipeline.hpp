#pragma once

#include "utils/optix/module.hpp"
#include "utils/optix/program_group.hpp"
#include "utils/optix/pipeline.hpp"
#include "utils/optix/sbt_record.hpp"

#include "mwir/modules/render_module.h"
#include "mwir/context.hpp"

namespace MWIR
{

class ForwardPipeline
{

public:
    ForwardPipeline();
    ~ForwardPipeline();
    ForwardPipeline(const ForwardPipeline&) = delete;
    ForwardPipeline& operator=(const ForwardPipeline&) = delete;
    ForwardPipeline(ForwardPipeline&&) = default;
    ForwardPipeline& operator=(ForwardPipeline&&) = default;

    void Render(int n_rays, int batch_size);
    
protected:
    friend class ForwardRendererImpl;

    std::unique_ptr<OptiX::Module> module;
    std::unique_ptr<OptiX::ProgramGroup> raygen_prog_group;
    std::unique_ptr<OptiX::ProgramGroup> miss_prog_group;
    std::unique_ptr<OptiX::ProgramGroup> hit_prog_group;
    std::unique_ptr<OptiX::Pipeline> pipeline;
    std::unique_ptr<OptiX::SBTRecord<RayGenData>> raygen_record;
    std::unique_ptr<OptiX::SBTRecord<MissData>> miss_record;
    std::unique_ptr<OptiX::SBTRecord<HitData>> hit_record;  
    OptixShaderBindingTable sbt;
};

}