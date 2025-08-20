#include "many_worlds_render_module.h"

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