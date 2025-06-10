#include <optix.h>
#include <optix_types.h>

#include "utils.h"
#include "vec_math.h"
#include "render_module.h"

extern "C"
{
    __constant__ Params params;
}

static __forceinline__ __device__ void setPayload( float3 p )
{
    optixSetPayload_0( __float_as_uint( p.x ) );
    optixSetPayload_1( __float_as_uint( p.y ) );
    optixSetPayload_2( __float_as_uint( p.z ) );
}

static __forceinline__ __device__ void computeRay( uint3 idx, uint3 dim, float3& origin, float3& direction )
{
    const float3 U = params.camera_u;
    const float3 V = params.camera_v;
    const float3 W = params.camera_w;
    const float2 d = 2.0f * make_float2(static_cast<float>( idx.x ) / static_cast<float>( dim.x ),
                                        static_cast<float>( idx.y ) / static_cast<float>( dim.y )) - 1.0f;
    origin    = params.camera_position;
    direction = normalize(d.x * U + d.y * V + W);
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    float3 ray_origin, ray_direction;
    computeRay(idx, dim, ray_origin, ray_direction);

    unsigned int p0;
    unsigned int p1;
    unsigned int p2;
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
            p0,
            p1,
            p2);
    float3 result;
    result.x = __uint_as_float( p0 );
    result.y = __uint_as_float( p1 );
    result.z = __uint_as_float( p2 );

    params.image[idx.y * params.image_width + idx.x] = make_color(result);
}

extern "C" __global__ void __miss__ms()
{
    setPayload(make_float3(0.5f, 0.6f, 0.9f));
}

extern "C" __global__ void __closesthit__ch()
{
    float3 vertices[3] = {};
    optixGetTriangleVertexData(optixGetGASTraversableHandle(), optixGetPrimitiveIndex(), optixGetSbtGASIndex(), 0, vertices );
    float3 normal = normalize( cross( vertices[1] - vertices[0], vertices[2] - vertices[0] ) );

    float ambientStrength = 0.5f;
    float3 ambient = ambientStrength * params.light_color;

    float diff = max(dot(normal, params.light_direction), 0.0);
    float3 diffuse = diff * params.light_color;

    float3 light = ambient + diffuse;

    float specularStrength = 0.5;
    if(dot(params.light_direction, normal) < 0.0) 
    {
        float3 world_position = optixGetWorldRayOrigin() + optixGetWorldRayDirection() * optixGetRayTmax();
        float3 viewDir = normalize(params.camera_position - world_position);
        float3 reflectDir = reflect(params.light_direction, normal);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 1);
        float3 specular = specularStrength * spec * params.light_color;
        light += specular;
    }

    setPayload(light);
}

