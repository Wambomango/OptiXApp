#include <optix.h>
#include <optix_types.h>

#include "vec_math.h"
#include "render_module.h"

extern "C"
{
    __constant__ Params params;
}

static __forceinline__ __device__ void computeRay( uint3 idx, uint3 dim, float3& origin, float3& direction )
{
    const float3 U = params.camera_u;
    const float3 V = params.camera_v;
    const float3 W = params.camera_w;
    const float2 d = 2.0f * make_float2(static_cast<float>( idx.x ) / static_cast<float>( dim.x ),
                                        static_cast<float>( idx.y ) / static_cast<float>( dim.y )) - 1.0f;
    origin    = params.camera_position;
    direction = normalize( d.x * U + d.y * V + W );
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    float3 ray_origin, ray_direction;
    computeRay(idx, dim, ray_origin, ray_direction);

    
    unsigned int p0;
    optixTrace(
            params.handle,
            ray_origin,
            ray_direction,
            0.0f,          
            1e16f,         
            0.0f, 
            OptixVisibilityMask( 255 ),
            OPTIX_RAY_FLAG_NONE,
            0,                  
            0,     
            0,              
            p0);



    params.image[idx.y * params.image_width + idx.x] = p0;
}




extern "C" __global__ void __miss__ms()
{
    MissData* miss_data  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    optixSetPayload_0(0);
}

extern "C" __global__ void __closesthit__ch()
{
    // When built-in triangle intersection is used, a number of fundamental
    // attributes are provided by the OptiX API, indlucing barycentric coordinates.
    const float2 barycentrics = optixGetTriangleBarycentrics();
    optixSetPayload_0(255);
}

