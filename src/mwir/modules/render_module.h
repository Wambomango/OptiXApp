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
    complex3 *result;
};

struct alignas(16) ManyWorldsParams
{
    float3 min;
    float3 max;
    float resolution;
    int n_samples;
    int3 shape;
    bool backward;

    float *occupancy;
    float3 *normal;

    complex3 *reference;
    complex3 *perturbation;

    OptixTraversableHandle mesh_handle;
};

struct alignas(16) Params
{    
    SceneParams scene;
    ManyWorldsParams many_worlds;

    int antenna_index;

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