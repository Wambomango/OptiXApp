#include <optix.h>
#include <optix_types.h>

#include "utils.h"
#include "vec_math.h"
#include "render_module.h"

extern "C"
{
    __constant__ Params params;
}


void __forceinline__ __device__ SetMiss()
{
    unsigned int bitmask = optixGetPayload_0();
    bitmask &= ~(1 << 0); // Clear the first bit
    optixSetPayload_0(bitmask);
}

static __forceinline__ __device__ void SetHit()
{
    unsigned int bitmask = optixGetPayload_0();
    bitmask |= 1 << 0; // Set the first bit
    optixSetPayload_0(bitmask);
}

void __forceinline__ __device__ SetPayload(const float3& position, const float3& normal)
{
    unsigned int x = __float_as_uint(position.x);
    unsigned int y = __float_as_uint(position.y);
    unsigned int z = __float_as_uint(position.z);
    unsigned int nx = __float_as_uint(normal.x);
    unsigned int ny = __float_as_uint(normal.y);
    unsigned int nz = __float_as_uint(normal.z);

    optixSetPayload_1(x);
    optixSetPayload_2(y);
    optixSetPayload_3(z);
    optixSetPayload_4(nx);
    optixSetPayload_5(ny);
    optixSetPayload_6(nz);
}

static __forceinline__ __device__ void computeRay(uint3 idx, uint3 dim, float3& origin, float3& direction )
{
    AntennaData sender = params.d_senders[params.antenna_index];
    
    float factor_x = (params.grid_x * OPTIX_MAX_GRID_DIM + static_cast<float>( idx.x )) / static_cast<float>(sender.n_rays.x) - 0.5f;


    float azimuth = sender.fov.x * factor_x;

    float factor_y = (params.grid_y * OPTIX_MAX_GRID_DIM + static_cast<float>( idx.y )) / static_cast<float>(sender.n_rays.y)- 0.5f;
    float elevation = sender.fov.y * factor_y;

    // float sin_elevation = sin(sender.fov.y / 2) * 2 * (0.5f * (static_cast<float>( idx.y ) / static_cast<float>( dim.y - 1)) - 1.0f);
    // float elevation = asin(sin_elevation);
    // float elevation = sender.fov.y * ((params.grid_y * OPTIX_MAX_GRID_DIM + static_cast<float>( idx.y ) / static_cast<float>(sender.n_rays.y)) - 0.5f);
    float3 dir = make_float3(cos(azimuth) * cos(elevation), sin(azimuth) *  cos(elevation), sin(elevation)); // direction in sensor coordinates

    origin = sender.position;
    // direction = sender.forward * dir.x + sender.left * dir.y + sender.up * dir.z; // direction in world coordinates via matmul
    direction = dir;
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    float3 ray_origin, ray_direction;
    computeRay(idx, dim, ray_origin, ray_direction);

    unsigned int bitmask = 0;
    unsigned int x_hit = 0, y_hit = 0, z_hit = 0;
    unsigned int nx_hit = 0, ny_hit = 0, nz_hit = 0;
    optixTrace( params.mesh_handle,
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
                bitmask,
                x_hit,
                y_hit,
                z_hit,
                nx_hit,
                ny_hit,
                nz_hit);

    float3 p_hit = make_float3(__uint_as_float( x_hit ), __uint_as_float( y_hit ), __uint_as_float( z_hit ));    
    float3 n_hit = make_float3(__uint_as_float( nx_hit ), __uint_as_float( ny_hit ), __uint_as_float( nz_hit ));

    int ray_offset = idx.x * OPTIX_MAX_GRID_DIM * params.n_receivers * params.signal.n_frequencies +
                     idx.y * params.n_receivers * params.signal.n_frequencies;

    // Hit surface
    if( (bitmask & (1 << 0)) && dot(n_hit, ray_direction) < 0.0f)
    {
        for(int i = 0; i < params.n_receivers; i++)
        {
            int receiver_offset = ray_offset + i * params.signal.n_frequencies;
            
            AntennaData receiver = params.d_receivers[i];
            float3 dir = normalize(receiver.position - p_hit);
            
            if(dot(dir, n_hit) <= 0.0f)
            {
                continue; 
            }

            unsigned int los_bitmask = 0;
            optixTrace( params.mesh_handle,
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
                        los_bitmask);

            if( (los_bitmask & (1 << 0)) == 0 )
            {
                continue; // No line of sight to the receiver
            }

            for(int j = 0; j < params.signal.n_frequencies; j++)
            {
                int result_index = receiver_offset + j;
                params.result[result_index].x_re += 1.0f;
                params.result[result_index].y_re += 1.0f;
                params.result[result_index].z_re += 1.0f;
            }
        }
    }
}

extern "C" __global__ void __miss__ms()
{
    SetMiss();
}

extern "C" __global__ void __closesthit__ch()
{
    float3 position = optixGetWorldRayOrigin() + optixGetWorldRayDirection() * optixGetRayTmax();

    float3 vertices[3] = {};
    optixGetTriangleVertexData(optixGetGASTraversableHandle(), optixGetPrimitiveIndex(), optixGetSbtGASIndex(), 0, vertices );
    float3 normal = normalize( cross( vertices[1] - vertices[0], vertices[2] - vertices[0] ) );


    SetHit();
    SetPayload(position, normal);
}

