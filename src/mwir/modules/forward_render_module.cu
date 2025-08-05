#include <optix.h>
#include <optix_types.h>
#include <curand_kernel.h>

#include "utils.h"
#include "vec_math.h"
#include "render_module.h"
#include "complex.h"

extern "C"
{
    __constant__ Params params;
}

static __device__ void CalculateE(uint3 idx, float3 dir_tx, float3 p_hit, float3 n_hit, complex3* result)
{
    int ray_offset = idx.x * OPTIX_MAX_GRID_DIM * params.scene.n_receivers * params.scene.signal.n_samples +
                    idx.y * params.scene.n_receivers * params.scene.signal.n_samples;
    int receiver_offset;

    AntennaData sender = params.scene.d_senders[params.antenna_index];
    float3 pos_tx = sender.position;
    float dist_tx = length(p_hit - pos_tx);

    AntennaData receiver;
    unsigned int p0;
    float3 dir_rx;
    float dist_rx;
    float dist_total;
    complex minusjomega;

    float factor = dist_tx / (2 * PI * C0 * sender.ray_density * dot(-dir_tx, n_hit));
    float3 vec_tx = factor * cross(n_hit, cross(dir_tx, normalize(cross(dir_tx, sender.left))));

    float3 vec_rx;
    complex3 A_rx;
    complex3 E_rx;
    for(int i = 0; i < params.scene.n_receivers; i++)
    {
        receiver_offset = ray_offset + i * params.scene.signal.n_samples;
        receiver = params.scene.d_receivers[i];
        dir_rx = normalize(receiver.position - p_hit);
                
        if(dot(dir_rx, n_hit) <= 0.0f)
        {
            continue; 
        }
        
        dist_rx = length(receiver.position - p_hit);
        optixTrace( params.scene.mesh_handle,
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

        if(__uint_as_float(p0) > 0.0f)
        {
            continue; 
        }
        
        dist_total = dist_tx + dist_rx;
        vec_rx = vec_tx / dist_rx;

        for(int j = 0; j < params.scene.signal.n_samples; j++)
        {
            minusjomega = make_complex(0.0f, -(params.scene.signal.frequency_range.x + j * params.scene.signal.f_step));
            A_rx = vec_rx * expf(minusjomega * INV_C0 * dist_total);
            E_rx = minusjomega * A_rx;
            E_rx = E_rx - dot(E_rx, dir_rx) * dir_rx;
            result[receiver_offset + j] += E_rx;
        }
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

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    curandState rand_state;
    curand_init(params.seed + params.antenna_index, idx.x * dim.y + idx.y, 0, &rand_state);

    AntennaData sender = params.scene.d_senders[params.antenna_index];
    float3 p_tx = sender.position;
    float3 dir_tx;

    for(int i = 0; i < sender.n_batches; i++)
    {   
        dir_tx = SampleDir(sender, rand_state);
        optixTrace( params.scene.mesh_handle,
                    p_tx,
                    dir_tx,
                    0.0f,          
                    1e16f,         
                    0.0f, 
                    OptixVisibilityMask( 255 ),
                    OPTIX_RAY_FLAG_NONE,
                    0,                  
                    0,     
                    0);
    }
}

extern "C" __global__ void __miss__geometry()
{
    optixSetPayload_0(1);
}

extern "C" __global__ void __closesthit__geometry()
{
    uint3 idx = optixGetLaunchIndex();
    uint3 dim = optixGetLaunchDimensions();

    float3 p_hit = optixGetWorldRayOrigin() + optixGetWorldRayDirection() * optixGetRayTmax();
    float3 vertices[3] = {};
    optixGetTriangleVertexData(optixGetGASTraversableHandle(), optixGetPrimitiveIndex(), optixGetSbtGASIndex(), 0, vertices );
    float3 n_hit = normalize( cross( vertices[1] - vertices[0], vertices[2] - vertices[0] ) );

    if(dot(n_hit, optixGetWorldRayDirection()) >= 0.0f)
    {
        return;
    }

    CalculateE(idx, optixGetWorldRayDirection(), p_hit, n_hit, params.scene.result);
}


extern "C" __global__ void __miss__antenna()
{
    optixSetPayload_0(0);
}

extern "C" __global__ void __closesthit__antenna()
{
    optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
}