#pragma once

#include <vector_types.h>
#include <curand_kernel.h>

#include "common.h"
#include "complex.h"
#include "vec_math.h"
#include "defines.h"

__forceinline__ __device__ float3 exp( const float3& x )
{
    return make_float3( exp( x.x ), exp( x.y ), exp( x.z ) );
}

static __forceinline__ __device__ int LinearizeIndex(const int &x, const int &y, const int &z, const int3 &shape)
{
    return x * shape.y * shape.z + y * shape.z + z;
}

static __forceinline__ __device__ int GetRayOffset(const uint3 &idx, const Params &p)
{
    return idx.x * OPTIX_MAX_GRID_DIM * p.scene.n_receivers * p.scene.signal.n_samples + idx.y * p.scene.n_receivers * p.scene.signal.n_samples;
}

static __forceinline__ __device__ float3 SafeNormalize(const float3 &v, curandState &rand_state)
{
    float len = length(v);
    if (len > NUMERICAL_EPS)
    {
        return v / len;
    }
    else 
    {
        float u = curand_uniform(&rand_state); 
        float v = curand_uniform(&rand_state); 
        float theta = 2.0f * PI * u;
        float phi = acosf(2.0f * v - 1.0f);
        float x = sinf(phi) * cosf(theta);
        float y = sinf(phi) * sinf(theta);
        float z = cosf(phi);
        return make_float3(x, y, z);
    }
}

static __forceinline__ __device__ float3 SampleDir(const AntennaData& sender, curandState& rand_state)
{
    float u = curand_uniform(&rand_state);
    float v = curand_uniform(&rand_state);
    float azimuth = sender.fov.x * (u - 0.5f);
    float elevation = asin(sin(sender.fov.y / 2) * (2 * v - 1.0f));
    float3 dir = make_float3(cos(azimuth) * cos(elevation), sin(azimuth) * cos(elevation), sin(elevation));
    return sender.forward * dir.x + sender.left * dir.y + sender.up * dir.z;
}


static __device__ void CalculateE(const Params &p, const int &ray_offset, const float3 &dir_tx, const float3 &p_hit, const float3 &n_hit, complex3* const result, const bool overwrite)
{
    AntennaData sender = p.scene.d_senders[p.antenna_index];
    float3 pos_tx = sender.position;
    float dist_tx = length(p_hit - pos_tx);
    for(int i = 0; i < p.scene.n_receivers; i++)
    {
        AntennaData receiver = p.scene.d_receivers[i];
        float3 dir_rx = normalize(receiver.position - p_hit);
        float dist_rx = length(receiver.position - p_hit);        
             
        bool success = false;
        if(dot(dir_rx, n_hit) > 0.0f)
        {

            unsigned int p0;
            optixTrace( p.scene.mesh_handle,
                        p_hit + n_hit * 0.001f, 
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
            success = (__uint_as_float(p0) <= 0.0f); 
        }

        if(success)
        {
            for(int j = 0; j < p.scene.signal.n_samples; j++)
            {
                int offset = ray_offset + i * p.scene.signal.n_samples + j;
                complex minusjomega = make_complex(0.0f, -(p.scene.signal.frequency_range.x + j * p.scene.signal.f_step));
                complex3 Epsilon_rx = minusjomega * (dist_tx / (2 * PI * C0 * sender.ray_density * dot(-dir_tx, n_hit) * dist_rx)) * cross(n_hit, cross(dir_tx, normalize(cross(dir_tx, sender.left)))) * expf(minusjomega * INV_C0 * (dist_tx + dist_rx));
                complex3 E_rx = Epsilon_rx - dot(Epsilon_rx, dir_rx) * dir_rx;
                result[offset] = (overwrite) ? (E_rx) : (result[offset] + E_rx);
            }
        }
        else if(overwrite)
        {
            for(int j = 0; j < p.scene.signal.n_samples; j++)
            {
                int offset = ray_offset + i * p.scene.signal.n_samples + j;
                result[offset] = make_complex3(0.0f);
            }
        }
    }
}

