#pragma once

#include "complex.h"
#include "types.h"
#include "defines.h"
#include <curand_kernel.h>
#include <optix_types.h>

struct alignas(16) SceneParams
{
    OptixTraversableHandle mesh_handle;
    unsigned int n_senders;
    AntennaData *h_senders;
    AntennaData *d_senders;
    unsigned int n_receivers;
    AntennaData *h_receivers;
    AntennaData *d_receivers;
    SignalData signal;
};


struct alignas(16) ManyWorldsParams
{
    float3 min;
    float3 max;
    float resolution;
    int n_samples;
    int3 shape;

    float *occupancy;
    float3 *normal;
};


struct alignas(16) Params
{    
    SceneParams scene;
    ManyWorldsParams many_worlds;

    int antenna_index;
    complex3 *result;

    int seed;
    curandState *randstates;
};

struct RayGenData
{
};

struct MissData
{
};

struct HitData
{
};