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

static __forceinline__ __device__ float3 SampleDir(const AntennaData& sender, curandState& rand_state)
{
    float u = curand_uniform(&rand_state);
    float v = curand_uniform(&rand_state);
    float azimuth = sender.fov.x * (u - 0.5f);
    float elevation = asin(sin(sender.fov.y / 2) * (2 * v - 1.0f));
    float3 dir = make_float3(cos(azimuth) * cos(elevation), sin(azimuth) * cos(elevation), sin(elevation));
    return sender.forward * dir.x + sender.left * dir.y + sender.up * dir.z;
}

static __forceinline__ __device__ int LinearizeIndex(int x, int y, int z)
{
    return x * params.many_worlds.shape.y * params.many_worlds.shape.z + y * params.many_worlds.shape.z + z;
}

static __forceinline__ __device__ float SampleOccupancy(float3 &normalized_idx)
{
    float3 continuous_idx = make_float3(static_cast<int>(normalized_idx.x * params.many_worlds.shape.x + 0.5f),
                                        static_cast<int>(normalized_idx.y * params.many_worlds.shape.y + 0.5f),
                                        static_cast<int>(normalized_idx.z * params.many_worlds.shape.z + 0.5f));

    int3 idx = make_int3(static_cast<int>(roundf(continuous_idx.x)),
                        static_cast<int>(roundf(continuous_idx.y)),
                        static_cast<int>(roundf(continuous_idx.z)));

    int idx_linear = LinearizeIndex(idx.x, idx.y, idx.z);
    atomicAdd(&params.many_worlds.occupancy[idx_linear], 0.1f);
    return 0.0f;
}

static __forceinline__ __device__ float3 SampleNormal(float3 &idx)
{

    return make_float3(0.0f, 0.0f, 1.0f); 
}

static __forceinline__ __device__ void ManyWorldsContribution(float3 &p_tx, float3 &dir_tx, float &t_sample)
{
    float3 p_sample = p_tx + dir_tx * t_sample;
    float3 normalized_idx = make_float3((p_sample.x - params.many_worlds.min.x) / (params.many_worlds.resolution * params.many_worlds.shape.x),
                                        (p_sample.y - params.many_worlds.min.y) / (params.many_worlds.resolution * params.many_worlds.shape.y),
                                        (p_sample.z - params.many_worlds.min.z) / (params.many_worlds.resolution * params.many_worlds.shape.z));

    if(normalized_idx.x < 0 || normalized_idx.x > 1 ||
       normalized_idx.y < 0 || normalized_idx.y > 1 ||
       normalized_idx.z < 0 || normalized_idx.z > 1)
    {
        return; // Out of bounds
    }

    float occupancy = SampleOccupancy(normalized_idx);
    float3 normal = SampleNormal(normalized_idx);



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
    unsigned int p0;
    unsigned int p1;
    float t_bb0;
    float t_bb1;
    float t_sample;

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

        p0 = __float_as_uint(-1.0f);
        p1 = __float_as_uint(-1.0f);
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

        t_bb0 = __uint_as_float( p0 );
        t_bb1 = __uint_as_float( p1 );

        if(t_bb0 < 0.0f)
        {
            continue;
        }

        for(int j = 0; j < params.many_worlds.n_samples; j++)
        {
            if(t_bb1 < 0.0f)
            {
                t_sample = t_bb0 * curand_uniform(&rand_state);
            }
            else
            {
                t_sample = t_bb0 + (t_bb1 - t_bb0) * curand_uniform(&rand_state);
            }

            ManyWorldsContribution(p_tx, dir_tx, t_sample);
        }
    }
}





extern "C" __global__ void __miss__geometry()
{
    optixSetPayload_0(1);
}
extern "C" __global__ void __closesthit__geometry()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    float3 p_hit = optixGetWorldRayOrigin() + optixGetWorldRayDirection() * optixGetRayTmax();
    float3 vertices[3] = {};
    optixGetTriangleVertexData(optixGetGASTraversableHandle(), optixGetPrimitiveIndex(), optixGetSbtGASIndex(), 0, vertices );
    float3 n_hit = normalize( cross( vertices[1] - vertices[0], vertices[2] - vertices[0] ) );

    if(dot(n_hit, optixGetWorldRayDirection()) >= 0.0f)
    {
        return;
    }

    int ray_offset = idx.x * OPTIX_MAX_GRID_DIM * params.scene.n_receivers * params.scene.signal.n_samples +
                    idx.y * params.scene.n_receivers * params.scene.signal.n_samples;
    int receiver_offset;

    AntennaData sender = params.scene.d_senders[params.antenna_index];
    float3 pos_tx = sender.position;
    float3 dir_tx = optixGetWorldRayDirection();
    float dist_tx = length(p_hit - pos_tx);

    AntennaData receiver;
    unsigned int bitmask;
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
        
        bitmask = 0;
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
                    bitmask);


        if(bitmask == 0)
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
            params.result[receiver_offset + j] += E_rx;
        }
    }
}


extern "C" __global__ void __miss__antenna()
{
    optixSetPayload_0(1);
}
extern "C" __global__ void __closesthit__antenna()
{
}




extern "C" __global__ void __miss__manyworlds()
{
}
extern "C" __global__ void __closesthit__manyworlds()
{ 
    unsigned int p0 = optixGetPayload_0();
    unsigned int p1 = optixGetPayload_1();;
    float t_hit = optixGetRayTmax();

    if(__uint_as_float( p0 ) < 0.0f)
    {
        float3 p_tx = optixGetWorldRayOrigin();
        float3 dir_tx = optixGetWorldRayDirection();

        p0 = __float_as_uint(t_hit);
        optixTrace( params.many_worlds.mesh_handle,
                    p_tx,
                    dir_tx,
                    t_hit + 0.0001f,          
                    1e16f,         
                    0.0f, 
                    OptixVisibilityMask( 255 ),
                    OPTIX_RAY_FLAG_NONE,
                    2,                  
                    0,     
                    2,
                    p0,
                    p1);

        optixSetPayload_0(p0);
        optixSetPayload_1(p1);
    }
    else
    {
        p1 = __float_as_uint(t_hit);
        optixSetPayload_1(p1);
    }
}



