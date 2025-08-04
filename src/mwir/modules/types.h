#pragma once

#include "vec_math.h"

struct alignas(16) AntennaData
{
    float3 position;
    float3 forward;
    float3 left;
    float3 up;
    float2 fov;
    float solid_angle;
    float ray_density;
    int n_rays;
    int n_batches;
};

struct alignas(16) SignalData
{
    float2 frequency_range;
    int n_samples;
    float f_step;
};
