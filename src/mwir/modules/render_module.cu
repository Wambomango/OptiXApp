#include "render_module.h"

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
    float3 p_hit = optixGetWorldRayOrigin() + optixGetWorldRayDirection() * optixGetRayTmax();
    float3 vertices[3] = {};
    optixGetTriangleVertexData(optixGetGASTraversableHandle(), optixGetPrimitiveIndex(), optixGetSbtGASIndex(), 0, vertices );
    float3 n_hit = normalize( cross( vertices[1] - vertices[0], vertices[2] - vertices[0] ) );
    if(dot(n_hit, optixGetWorldRayDirection()) >= 0.0f)
    {
        return;
    }
    CalculateE(params, idx, optixGetWorldRayDirection(), p_hit, n_hit, params.scene.result, false);
}







extern "C" __global__ void __miss__antenna()
{
    optixSetPayload_0(0);
}
extern "C" __global__ void __closesthit__antenna()
{
    optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
}