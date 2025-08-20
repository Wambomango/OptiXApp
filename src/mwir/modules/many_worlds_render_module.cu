#include "many_worlds_render_module.h"


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

    float3 n00 = SafeNormalize(n000 * (1 - deltas.x) + n001 * deltas.x);    // Maybe safe normalize?
    float3 n01 = SafeNormalize(n010 * (1 - deltas.x) + n011 * deltas.x);
    float3 n10 = SafeNormalize(n100 * (1 - deltas.x) + n101 * deltas.x);
    float3 n11 = SafeNormalize(n110 * (1 - deltas.x) + n111 * deltas.x);
    float3 n0 = SafeNormalize(n00 * (1 - deltas.y) + n01 * deltas.y);
    float3 n1 = SafeNormalize(n10 * (1 - deltas.y) + n11 * deltas.y);
    normal = SafeNormalize(n0 * (1 - deltas.z) + n1 * deltas.z);
}

static __forceinline__ __device__ void AddPerturbation(const uint3 &idx, const float3 &p_tx, const float3 &dir_tx, const float &t_sample)
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
        complex3 *reference = params.many_worlds.reference;
        complex3 *result = params.scene.result;

        int ray_offset = idx.x * OPTIX_MAX_GRID_DIM * params.scene.n_receivers * params.scene.signal.n_samples + idx.y * params.scene.n_receivers * params.scene.signal.n_samples;
        int receiver_offset;
        for(int i = 0; i < params.scene.n_receivers; i++)
        {
            receiver_offset = ray_offset + i * params.scene.signal.n_samples;
            for(int j = 0; j < params.scene.signal.n_samples; j++)
            {
                result[receiver_offset + j] += params.many_worlds.weight * reference[receiver_offset + j];
            }
        }
    }
    else
    {
        CalculateE(params, idx, dir_tx, p_sample, normal, params.many_worlds.perturbation, true); 

        complex3 *reference = params.many_worlds.reference;
        complex3 *perturbation = params.many_worlds.perturbation;
        complex3 *result = params.scene.result;
        int ray_offset = idx.x * OPTIX_MAX_GRID_DIM * params.scene.n_receivers * params.scene.signal.n_samples + idx.y * params.scene.n_receivers * params.scene.signal.n_samples;
        int receiver_offset;

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

            AddPerturbation(idx, p_tx, dir_tx, t_sample);
        }
    }
}



extern "C" __global__ void __raygen__rg()
{
    uint3 idx = optixGetLaunchIndex();
    uint3 dim = optixGetLaunchDimensions();
    curandState rand_state;
    curand_init(params.seed + params.antenna_index, idx.x * dim.y + idx.y, 0, &rand_state);

    AntennaData sender = params.scene.d_senders[params.antenna_index];
    float3 p_tx = sender.position;
    float3 dir_tx;
    unsigned int p0;
    float t_hit;

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
                    0,
                    p0);
        t_hit = __uint_as_float(p0);
        PerturbRay(idx, p_tx, dir_tx, t_hit, rand_state);
    }
}



extern "C" __global__ void __miss__geometry()
{
    optixSetPayload_0(__float_as_uint(1E16f));
    uint3 idx = optixGetLaunchIndex();
    complex3 *reference = params.many_worlds.reference;
    int ray_offset = idx.x * OPTIX_MAX_GRID_DIM * params.scene.n_receivers * params.scene.signal.n_samples + idx.y * params.scene.n_receivers * params.scene.signal.n_samples;
    int receiver_offset;
    for(int i = 0; i < params.scene.n_receivers; i++)
    {
        receiver_offset = ray_offset + i * params.scene.signal.n_samples;
        for(int j = 0; j < params.scene.signal.n_samples; j++)
        {
            reference[receiver_offset + j] = make_complex3(0.0f);
        }
    }
}
extern "C" __global__ void __closesthit__geometry()
{
    optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
    uint3 idx = optixGetLaunchIndex();
    float3 p_hit = optixGetWorldRayOrigin() + optixGetWorldRayDirection() * optixGetRayTmax();
    float3 vertices[3] = {};
    optixGetTriangleVertexData(optixGetGASTraversableHandle(), optixGetPrimitiveIndex(), optixGetSbtGASIndex(), 0, vertices );
    float3 n_hit = normalize( cross( vertices[1] - vertices[0], vertices[2] - vertices[0] ) );
    if(dot(n_hit, optixGetWorldRayDirection()) >= 0.0f)
    {
        return;
    }
    CalculateE(params, idx, optixGetWorldRayDirection(), p_hit, n_hit, params.many_worlds.reference, true);
}



extern "C" __global__ void __miss__antenna()
{
    optixSetPayload_0(0);
}
extern "C" __global__ void __closesthit__antenna()
{
    optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
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