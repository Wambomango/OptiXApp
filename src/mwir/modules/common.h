#pragma once

#include "complex.h"

#include <optix.h>
#include <optix_types.h>
#include <curand_kernel.h>

struct alignas(16) AntennaData
{
    float3 position;
    float3 forward;
    float3 left;
    float3 up;
    float2 fov;
    float solid_angle;
    float ray_density;
    long n_rays;
    long n_batches;
};

struct alignas(16) SignalData
{
    float2 frequency_range;
    int n_samples;
    float f_step;
};

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
    complex3 *merged_result;
};

struct alignas(16) ManyWorldsParams
{
    float3 min;
    float3 max;
    float resolution;
    int n_samples;
    int3 shape;
    float weight;
    bool backward;

    complex3 *e_field_gradient;

    float *occupancy;
    float *occupancy_gradient;

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