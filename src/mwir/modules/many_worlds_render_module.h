#pragma once

#include "defines.h"
#include "utils.h"
#include "common.h"

extern "C"
{
    __constant__ Params params;
}

static __forceinline__ __device__ void SampleManyWorlds(const float3 &normalized_idx, float &occupancy, float3 &normal)
{
    float3 continuous_idx = make_float3(normalized_idx.x * params.many_worlds.shape.x - 0.5f,
                                        normalized_idx.y * params.many_worlds.shape.y - 0.5f,
                                        normalized_idx.z * params.many_worlds.shape.z - 0.5f);

    int3 lower_idx = make_int3(max(0, static_cast<int>(floorf(continuous_idx.x))),
                               max(0, static_cast<int>(floorf(continuous_idx.y))),
                               max(0, static_cast<int>(floorf(continuous_idx.z))));

    int3 upper_idx = make_int3( min(params.many_worlds.shape.x - 1, static_cast<int>(ceilf(continuous_idx.x))),
                                min(params.many_worlds.shape.y - 1, static_cast<int>(ceilf(continuous_idx.y))),
                                min(params.many_worlds.shape.z - 1, static_cast<int>(ceilf(continuous_idx.z))));
    
    float3 deltas = make_float3(continuous_idx.x - lower_idx.x,
                                continuous_idx.y - lower_idx.y,
                                continuous_idx.z - lower_idx.z);

    float *occ = params.many_worlds.occupancy;
    float o000 = occ[LinearizeIndex(lower_idx.x, lower_idx.y, lower_idx.z, params.many_worlds.shape)];
    float o001 = occ[LinearizeIndex(lower_idx.x, lower_idx.y, upper_idx.z, params.many_worlds.shape)];
    float o010 = occ[LinearizeIndex(lower_idx.x, upper_idx.y, lower_idx.z, params.many_worlds.shape)];
    float o011 = occ[LinearizeIndex(lower_idx.x, upper_idx.y, upper_idx.z, params.many_worlds.shape)];
    float o100 = occ[LinearizeIndex(upper_idx.x, lower_idx.y, lower_idx.z, params.many_worlds.shape)];
    float o101 = occ[LinearizeIndex(upper_idx.x, lower_idx.y, upper_idx.z, params.many_worlds.shape)];
    float o110 = occ[LinearizeIndex(upper_idx.x, upper_idx.y, lower_idx.z, params.many_worlds.shape)];
    float o111 = occ[LinearizeIndex(upper_idx.x, upper_idx.y, upper_idx.z, params.many_worlds.shape)];

    float3 *nrm = params.many_worlds.normal;
    float3 n000 = nrm[LinearizeIndex(lower_idx.x, lower_idx.y, lower_idx.z, params.many_worlds.shape)];
    float3 n001 = nrm[LinearizeIndex(lower_idx.x, lower_idx.y, upper_idx.z, params.many_worlds.shape)];
    float3 n010 = nrm[LinearizeIndex(lower_idx.x, upper_idx.y, lower_idx.z, params.many_worlds.shape)];
    float3 n011 = nrm[LinearizeIndex(lower_idx.x, upper_idx.y, upper_idx.z, params.many_worlds.shape)];
    float3 n100 = nrm[LinearizeIndex(upper_idx.x, lower_idx.y, lower_idx.z, params.many_worlds.shape)];
    float3 n101 = nrm[LinearizeIndex(upper_idx.x, lower_idx.y, upper_idx.z, params.many_worlds.shape)];
    float3 n110 = nrm[LinearizeIndex(upper_idx.x, upper_idx.y, lower_idx.z, params.many_worlds.shape)];
    float3 n111 = nrm[LinearizeIndex(upper_idx.x, upper_idx.y, upper_idx.z, params.many_worlds.shape)];

    float o00 = o000 * (1 - deltas.x) + o001 * deltas.x;
    float o01 = o010 * (1 - deltas.x) + o011 * deltas.x;
    float o10 = o100 * (1 - deltas.x) + o101 * deltas.x;
    float o11 = o110 * (1 - deltas.x) + o111 * deltas.x;
    float o0 = o00 * (1 - deltas.y) + o01 * deltas.y;
    float o1 = o10 * (1 - deltas.y) + o11 * deltas.y;
    occupancy = o0 * (1 - deltas.z) + o1 * deltas.z;

    float3 n00 = SafeNormalize(n000 * (1 - deltas.x) + n001 * deltas.x);
    float3 n01 = SafeNormalize(n010 * (1 - deltas.x) + n011 * deltas.x);
    float3 n10 = SafeNormalize(n100 * (1 - deltas.x) + n101 * deltas.x);
    float3 n11 = SafeNormalize(n110 * (1 - deltas.x) + n111 * deltas.x);
    float3 n0 = SafeNormalize(n00 * (1 - deltas.y) + n01 * deltas.y);
    float3 n1 = SafeNormalize(n10 * (1 - deltas.y) + n11 * deltas.y);
    normal = SafeNormalize(n0 * (1 - deltas.z) + n1 * deltas.z);
}


static __forceinline__ __device__ void AddPerturbationForward(const uint3 &idx, const float3 &p_tx, const float3 &dir_tx, const float &t_sample)
{
    float3 p_sample = p_tx + dir_tx * t_sample;
    float3 normalized_idx = make_float3((p_sample.x - params.many_worlds.min.x) / (params.many_worlds.resolution * params.many_worlds.shape.x),
                                        (p_sample.y - params.many_worlds.min.y) / (params.many_worlds.resolution * params.many_worlds.shape.y),
                                        (p_sample.z - params.many_worlds.min.z) / (params.many_worlds.resolution * params.many_worlds.shape.z));

    if(normalized_idx.x < 0 || normalized_idx.x > 1 ||
       normalized_idx.y < 0 || normalized_idx.y > 1 ||
       normalized_idx.z < 0 || normalized_idx.z > 1)
    {
        return;
    }

    float occupancy;
    float3 normal;
    SampleManyWorlds(normalized_idx, occupancy, normal);

    if(dot(normal, dir_tx) >= 0.0f)
    {
        // Backfacing surface patch, set occupancy to 0.
        occupancy = 0.0f;
    }
    else
    {
        // Forward facing surface patch, calculate the perturbation, blend between reference and perturbation
        CalculateE(params, idx, dir_tx, p_sample, normal, params.many_worlds.perturbation, true); 
    }
    
    int ray_offset = idx.x * OPTIX_MAX_GRID_DIM * params.scene.n_receivers * params.scene.signal.n_samples + idx.y * params.scene.n_receivers * params.scene.signal.n_samples;
    int receiver_offset;
    complex3 *reference = params.many_worlds.reference;
    complex3 *perturbation = params.many_worlds.perturbation;
    complex3 *result = params.scene.result;
    for(int i = 0; i < params.scene.n_receivers; i++)
    {
        receiver_offset = ray_offset + i * params.scene.signal.n_samples;
        for(int j = 0; j < params.scene.signal.n_samples; j++)
        {
            complex3 E_ref = reference[receiver_offset + j];
            complex3 E_pert = perturbation[receiver_offset + j];
            complex3 E_res = occupancy * E_pert + (1.0f - occupancy) * E_ref;
            result[receiver_offset + j] += params.many_worlds.weight * E_res;
        }
    }
}

static __forceinline__ __device__ void AddPerturbationBackward(const uint3 &idx, const float3 &p_tx, const float3 &dir_tx, const float &t_sample)
{














}

static __forceinline__ __device__ void PerturbRay(const uint3 &idx, const float3 &p_tx, const float3 &dir_tx, float &t_hit, curandState &rand_state)
{
    unsigned int p0 = __float_as_uint(-1.0f);
    unsigned int p1 = __float_as_uint(-1.0f);
    optixTrace( params.many_worlds.mesh_handle,
                p_tx,
                dir_tx,
                0.0f,          
                1e16f,         
                0.0f, 
                OptixVisibilityMask( 255 ),
                OPTIX_RAY_FLAG_NONE,
                2,                  
                0,     
                2,
                p0,
                p1);
    float t_bb0 = __uint_as_float( p0 );
    float t_bb1 = __uint_as_float( p1 );

    if(t_bb0 < 0.0f)
    {
        // Many Worlds was not hit
        complex3 *reference = params.many_worlds.reference;
        complex3 *result = params.scene.result;
        int ray_offset = idx.x * OPTIX_MAX_GRID_DIM * params.scene.n_receivers * params.scene.signal.n_samples + idx.y * params.scene.n_receivers * params.scene.signal.n_samples;
        int receiver_offset;
        for(int i = 0; i < params.scene.n_receivers; i++)
        {
            receiver_offset = ray_offset + i * params.scene.signal.n_samples;
            for(int j = 0; j < params.scene.signal.n_samples; j++)
            {
                result[receiver_offset + j] +=  params.many_worlds.weight * reference[receiver_offset + j];
            }
        }
    }
    else
    {
        // Many Worlds was hit
        float t_sample;
        for(int j = 0; j < params.many_worlds.n_samples; j++)
        {
            if(t_bb1 < 0.0f)
            {
                t_sample = min(t_bb0, t_hit) * curand_uniform(&rand_state);
            }
            else
            {
                t_sample = t_bb0 + (min(t_bb1, t_hit) - t_bb0) * curand_uniform(&rand_state);
            }

            if(params.many_worlds.backward)
            {
                AddPerturbationBackward(idx, p_tx, dir_tx, t_sample);
            }
            else
            {
                AddPerturbationForward(idx, p_tx, dir_tx, t_sample);
            }
        }
    }
}
