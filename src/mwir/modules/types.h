#pragma once

struct AntennaData
{
    float3 position;
    float3 forward;
    float3 left;
    float3 up;
    float2 fov;
    float solid_angle;
    float ray_density;
    int2 n_rays;
};

struct SignalData
{
    float2 frequency_range;
    int n_frequencies;
    float f_step;
};

struct EField
{
    float x_re;
    float x_im;

    float y_re;
    float y_im;

    float z_re;
    float z_im;
};
