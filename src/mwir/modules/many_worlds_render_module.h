#pragma once

#include "defines.h"
#include "utils.h"
#include "common.h"

extern "C"
{
    __constant__ Params params;
}

static __forceinline__ __device__ void SampleManyWorlds(const float3 &normalized_idx, float &occupancy, float3 &occupancy_gradient)
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

    float o00 = o000 * (1 - deltas.x) + o100 * deltas.x;
    float o01 = o001 * (1 - deltas.x) + o101 * deltas.x;
    float o10 = o010 * (1 - deltas.x) + o110 * deltas.x;
    float o11 = o011 * (1 - deltas.x) + o111 * deltas.x;
    float o0 = o00 * (1 - deltas.y) + o10 * deltas.y;
    float o1 = o01 * (1 - deltas.y) + o11 * deltas.y;
    occupancy = o0 * (1 - deltas.z) + o1 * deltas.z;

    float grad_x = (o100 - o000) * (1 - deltas.y) * (1 - deltas.z) + (o101 - o001) * (1 - deltas.y) * deltas.z + (o110 - o010) * deltas.y * (1 - deltas.z) + (o111 - o011) * deltas.y * deltas.z;
    float grad_y = (o010 - o000) * (1 - deltas.x) * (1 - deltas.z) + (o011 - o001) * (1 - deltas.x) * deltas.z + (o110 - o100) * deltas.x * (1 - deltas.z) + (o111 - o101) * deltas.x * deltas.z;
    float grad_z = (o001 - o000) * (1 - deltas.x) * (1 - deltas.y) + (o011 - o010) * (1 - deltas.x) * deltas.y + (o101 - o100) * deltas.x * (1 - deltas.y) + (o111 - o110) * deltas.x * deltas.y;
    occupancy_gradient = make_float3(grad_x, grad_y, grad_z) * 1 / params.many_worlds.resolution;
}

static __forceinline__ __device__ void BackpropManyWorlds(const float3 &normalized_idx, float &occupancy_gradient, float3 &gradient_gradient)
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


    // Occupancy weights
    float ow000 = (1 - deltas.x) * (1 - deltas.y) * (1 - deltas.z);
    float ow001 = (1 - deltas.x) * (1 - deltas.y) * deltas.z;
    float ow010 = (1 - deltas.x) * deltas.y * (1 - deltas.z);
    float ow011 = (1 - deltas.x) * deltas.y * deltas.z;
    float ow100 = deltas.x * (1 - deltas.y) * (1 - deltas.z);
    float ow101 = deltas.x * (1 - deltas.y) * deltas.z;
    float ow110 = deltas.x * deltas.y * (1 - deltas.z);
    float ow111 = deltas.x * deltas.y * deltas.z;

    // Gradient weights 
    float3 gw000 = make_float3(-(1 - deltas.y) * (1 - deltas.z), -(1 - deltas.x) * (1 - deltas.z), -(1 - deltas.x) * (1 - deltas.y));
    float3 gw001 = make_float3(-(1 - deltas.y) * deltas.z, -(1 - deltas.x) * deltas.z, -(1 - deltas.x) * (1 - deltas.y));
    float3 gw010 = make_float3(-deltas.y * (1 - deltas.z), (1 - deltas.x) * (1 - deltas.z), -(1 - deltas.x) * deltas.y);
    float3 gw011 = make_float3(-deltas.y * deltas.z, (1 - deltas.x) * deltas.z, (1 - deltas.x) * deltas.y);
    float3 gw100 = make_float3((1 - deltas.y) * (1 - deltas.z), -deltas.x * (1 - deltas.z), -deltas.x * (1 - deltas.y));
    float3 gw101 = make_float3((1 - deltas.y) * deltas.z, -deltas.x * deltas.z, deltas.x * (1 - deltas.y));
    float3 gw110 = make_float3(deltas.y * (1 - deltas.z), deltas.x * (1 - deltas.z), -deltas.x * deltas.y);
    float3 gw111 = make_float3(deltas.y * deltas.z, deltas.x * deltas.z, deltas.x * deltas.y);

    atomicAdd(&params.many_worlds.occupancy_gradient[LinearizeIndex(lower_idx.x, lower_idx.y, lower_idx.z, params.many_worlds.shape)], ow000 * occupancy_gradient + dot(gw000, gradient_gradient) * 1 / params.many_worlds.resolution);
    atomicAdd(&params.many_worlds.occupancy_gradient[LinearizeIndex(lower_idx.x, lower_idx.y, upper_idx.z, params.many_worlds.shape)], ow001 * occupancy_gradient + dot(gw001, gradient_gradient) * 1 / params.many_worlds.resolution);
    atomicAdd(&params.many_worlds.occupancy_gradient[LinearizeIndex(lower_idx.x, upper_idx.y, lower_idx.z, params.many_worlds.shape)], ow010 * occupancy_gradient + dot(gw010, gradient_gradient) * 1 / params.many_worlds.resolution);
    atomicAdd(&params.many_worlds.occupancy_gradient[LinearizeIndex(lower_idx.x, upper_idx.y, upper_idx.z, params.many_worlds.shape)], ow011 * occupancy_gradient + dot(gw011, gradient_gradient) * 1 / params.many_worlds.resolution);
    atomicAdd(&params.many_worlds.occupancy_gradient[LinearizeIndex(upper_idx.x, lower_idx.y, lower_idx.z, params.many_worlds.shape)], ow100 * occupancy_gradient + dot(gw100, gradient_gradient) * 1 / params.many_worlds.resolution);
    atomicAdd(&params.many_worlds.occupancy_gradient[LinearizeIndex(upper_idx.x, lower_idx.y, upper_idx.z, params.many_worlds.shape)], ow101 * occupancy_gradient + dot(gw101, gradient_gradient) * 1 / params.many_worlds.resolution);
    atomicAdd(&params.many_worlds.occupancy_gradient[LinearizeIndex(upper_idx.x, upper_idx.y, lower_idx.z, params.many_worlds.shape)], ow110 * occupancy_gradient + dot(gw110, gradient_gradient) * 1 / params.many_worlds.resolution);
    atomicAdd(&params.many_worlds.occupancy_gradient[LinearizeIndex(upper_idx.x, upper_idx.y, upper_idx.z, params.many_worlds.shape)], ow111 * occupancy_gradient + dot(gw111, gradient_gradient) * 1 / params.many_worlds.resolution);
}

static __forceinline__ __device__ void AddPerturbationForward(const int &ray_offset, const float3 &p_tx, const float3 &dir_tx, const float &t_sample, curandState &rand_state)
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
    float3 occupancy_gradient;
    SampleManyWorlds(normalized_idx, occupancy, occupancy_gradient);

    float3 normal = SafeNormalize(-occupancy_gradient, rand_state);
    if(dot(normal, dir_tx) >= 0.0f)
    {
        // Backfacing surface patch, set occupancy to 0.
        occupancy = 0.0f;
    }
    else
    {
        // Forward facing surface patch, calculate the perturbation, blend between reference and perturbation
        CalculateE(params, ray_offset, dir_tx, p_sample, normal, params.many_worlds.perturbation, true); 
    }
    
    complex3 *reference = params.many_worlds.reference;
    complex3 *perturbation = params.many_worlds.perturbation;
    complex3 *result = params.scene.result;
    for(int i = 0; i < params.scene.n_receivers; i++)
    {
        int receiver_offset = ray_offset + i * params.scene.signal.n_samples;
        for(int j = 0; j < params.scene.signal.n_samples; j++)
        {
            int frequency_offset = receiver_offset + j;

            complex3 E_ref = reference[frequency_offset];
            complex3 E_pert = perturbation[frequency_offset];
            complex3 E_res = occupancy * E_pert + (1.0f - occupancy) * E_ref;
            result[frequency_offset] += params.many_worlds.weight * E_res;
        }
    }
}


static __forceinline__ __device__ void AddPerturbationBackward(const int &ray_offset, const float3 &p_tx, const float3 &dir_tx, const float &t_sample, curandState &rand_state)
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
    float3 occupancy_gradient;
    SampleManyWorlds(normalized_idx, occupancy, occupancy_gradient);

    float3 normal = SafeNormalize(-occupancy_gradient, rand_state);
    if(dot(normal, dir_tx) >= 0.0f)
    {
        // Backfacing surface patch does not propagate gradients
        return;
    }

    complex3 *reference = params.many_worlds.reference;
    complex3 *perturbation = params.many_worlds.perturbation;
    complex3 *result = params.scene.result;
    float partial_occupancy = 0.0f;
    float3 partial_normal = make_float3(0.0f, 0.0f, 0.0f);

    // ComputeE with gradients
    AntennaData sender = params.scene.d_senders[params.antenna_index];
    float3 pos_tx = sender.position;
    float dist_tx = length(p_sample - pos_tx);

    for(int i = 0; i < params.scene.n_receivers; i++)
    {
        int receiver_offset = ray_offset + i * params.scene.signal.n_samples;
        AntennaData receiver = params.scene.d_receivers[i];
        float3 dir_rx = normalize(receiver.position - p_sample);

        if(dot(dir_rx, normal) <= 0.0f)
        {
            continue; 
        }

        float dist_rx = length(receiver.position - p_sample);
        unsigned int p0;
        optixTrace( params.scene.mesh_handle,
                    p_sample + normal * 0.001f, 
                    dir_rx,
                    0.0f,          
                    dist_rx,         
                    0.0f, 
                    OptixVisibilityMask( 255 ),
                    OPTIX_RAY_FLAG_NONE,
                    1,                  
                    0,     
                    1,              
                    p0);

        if(__uint_as_float(p0) > 0.0f)
        {
            continue; 
        }
        
        for(int j = 0; j < params.scene.signal.n_samples; j++)
        {
            int frequency_offset = receiver_offset + j;
            complex minusjomega = make_complex(0.0f, -(params.scene.signal.frequency_range.x + j * params.scene.signal.f_step));
            complex factor = minusjomega * (dist_tx / (dist_rx * 2 * PI * C0 * sender.ray_density * dot(-dir_tx, normal))) * expf(minusjomega * INV_C0 * (dist_tx + dist_rx));
            float3 vector = cross(normal, cross(dir_tx, normalize(cross(dir_tx, sender.left))));
            complex3 Epsilon_rx = factor * vector;
            complex3 E_pert = Epsilon_rx - dot(Epsilon_rx, dir_rx) * dir_rx;
            complex3 E_ref = reference[frequency_offset];
            result[frequency_offset] += occupancy * E_pert + (1.0f - occupancy) * E_ref;

            // Occupancy gradient
            partial_occupancy += elsum(params.many_worlds.e_field_gradient[frequency_offset] * (params.many_worlds.weight * (E_pert - E_ref))).real;

            // Normal gradient
            complex3 dL_dE_rx_rt = occupancy * params.many_worlds.weight * params.many_worlds.e_field_gradient[frequency_offset];

            complex3 dL_dEpsilon =  dL_dE_rx_rt.x * (make_float3(1.0f, 0.0f, 0.0f) - dir_rx.x * dir_rx) + 
                                    dL_dE_rx_rt.y * (make_float3(0.0f, 1.0f, 0.0f) - dir_rx.y * dir_rx) + 
                                    dL_dE_rx_rt.z * (make_float3(0.0f, 0.0f, 1.0f) - dir_rx.z * dir_rx);

            float tmp0 = 1 / dot(dir_tx, normal);
            float3 tmp1 = cross(dir_tx, normalize(cross(dir_tx, sender.left)));
            partial_normal += -real(conj(dL_dEpsilon.x) * (factor * (vector.x * dir_tx * tmp0 + make_float3(0.0f, -tmp1.z, tmp1.y))) + 
                                    conj(dL_dEpsilon.y) * (factor * (vector.y * dir_tx * tmp0 + make_float3(tmp1.z, 0.0f, -tmp1.x))) + 
                                    conj(dL_dEpsilon.z) * (factor * (vector.z * dir_tx * tmp0 + make_float3(-tmp1.y, tmp1.x, 0.0f))));
        }
    }

    float3 partial_occupancy_gradient = -(partial_normal.x * (make_float3(1.0f, 0.0f, 0.0f) - normal.x * normal) + 
                                        partial_normal.y * (make_float3(0.0f, 1.0f, 0.0f) - normal.y * normal) + 
                                        partial_normal.z * (make_float3(0.0f, 0.0f, 1.0f) - normal.z * normal)) / max(length(occupancy_gradient), 0.01f);
    BackpropManyWorlds(normalized_idx, partial_occupancy, partial_occupancy_gradient);
}


static __forceinline__ __device__ void PerturbRay(const int &ray_offset, const float3 &p_tx, const float3 &dir_tx, float &t_hit, curandState &rand_state)
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
        for(int i = 0; i < params.scene.n_receivers; i++)
        {
            int receiver_offset = ray_offset + i * params.scene.signal.n_samples;
            for(int j = 0; j < params.scene.signal.n_samples; j++)
            {
                int frequency_offset = receiver_offset + j;
                result[frequency_offset] +=  params.many_worlds.weight * reference[frequency_offset];
            }
        }
    }
    else
    {
        // Many Worlds was hit
        for(int j = 0; j < params.many_worlds.n_samples; j++)
        {
            float t_sample;
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
                AddPerturbationBackward(ray_offset, p_tx, dir_tx, t_sample, rand_state);
            }
            else
            {
                AddPerturbationForward(ray_offset, p_tx, dir_tx, t_sample, rand_state);
            }
        }
    }
}
