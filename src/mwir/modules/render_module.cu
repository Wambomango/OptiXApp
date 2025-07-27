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

static __forceinline__ __device__ HField PropagatePhase(EField &efield, float distance, float wavenumber)
{
    HField hfield;
    float attenuation = 1 / (distance * distance);

    float phase = wavenumber * distance;
    float cos_phase = cos(phase);
    float sin_phase = sin(phase);

    hfield.x_re = attenuation * (efield.x_re * cos_phase - efield.x_im * sin_phase);
    hfield.x_im = attenuation * (efield.x_re * sin_phase + efield.x_im * cos_phase);

    hfield.y_re = attenuation * (efield.y_re * cos_phase - efield.y_im * sin_phase);
    hfield.y_im = attenuation * (efield.y_re * sin_phase + efield.y_im * cos_phase);

    hfield.z_re = attenuation * (efield.z_re * cos_phase - efield.z_im * sin_phase);
    hfield.z_im = attenuation * (efield.z_re * sin_phase + efield.z_im * cos_phase);

    return hfield;
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
    curand_init(params.antenna_index, idx.x * dim.y + idx.y, 0, &rand_state);

    AntennaData sender = params.d_senders[params.antenna_index];
    float3 p_tx = sender.position;
    float3 dir_tx;
    unsigned int bitmask = 0;    

    for(int i = 0; i < sender.n_batches; i++)
    {   
        dir_tx = SampleDir(sender, rand_state);

        optixTrace( params.mesh_handle,
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
                    bitmask);
    }
}

extern "C" __global__ void __miss__ms()
{
    optixSetPayload_0(1);
}

extern "C" __global__ void __closesthit__ch()
{
    float3 p_hit = optixGetWorldRayOrigin() + optixGetWorldRayDirection() * optixGetRayTmax();
    float3 vertices[3] = {};
    optixGetTriangleVertexData(optixGetGASTraversableHandle(), optixGetPrimitiveIndex(), optixGetSbtGASIndex(), 0, vertices );
    float3 n_hit = normalize( cross( vertices[1] - vertices[0], vertices[2] - vertices[0] ) );

    if(dot(n_hit, optixGetWorldRayDirection()) >= 0.0f)
    {
        return;
    }

    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    int ray_offset = idx.x * OPTIX_MAX_GRID_DIM * params.n_receivers * params.signal.n_frequencies +
                    idx.y * params.n_receivers * params.signal.n_frequencies;
    int receiver_offset;
    int frequency_offset;

    AntennaData sender = params.d_senders[params.antenna_index];
    float3 pos_tx = sender.position;
    float3 dir_tx = optixGetWorldRayDirection();
    float dist_tx = length(p_hit - pos_tx);


    (sender.ray_density * dist_tx) / (2 * M_PI * C0 * dot(-dir_tx, n_hit)); 
   




    // float factor_tx = MU0 * sender.ray_density / (2 * M_PI * dot(-dir_tx, n_hit));








    float3 dir_rx;
    float factor_rx;
    for(int i = 0; i < params.n_receivers; i++)
    {
        receiver_offset = ray_offset + i * params.signal.n_frequencies;
        AntennaData receiver = params.d_receivers[i];
        dir_rx = normalize(receiver.position - p_hit);
                
        if(dot(dir_rx, n_hit) <= 0.0f)
        {
            continue; 
        }

        unsigned int bitmask = 0;
        optixTrace( params.mesh_handle,
                    p_hit + n_hit * 0.001f, 
                    dir_rx,
                    0.0f,          
                    1e16f,         
                    0.0f, 
                    OptixVisibilityMask( 255 ),
                    OPTIX_RAY_FLAG_NONE,
                    0,                  
                    0,     
                    0,              
                    bitmask);

        if(bitmask == 0)
        {
            continue; 
        }


        // float dist_rx = length(receiver.position - p_hit);

        // float invariant_factor =   * 2 * dist_rx / dot(-dir_tx, n_hit) * sender.ray_density;


        for(int j = 0; j < params.signal.n_frequencies; j++)
        {
            // HField H_hit;
            // float frequency = params.signal.frequency_range.x + j * params.signal.f_step;
            int result_index = receiver_offset + j;
            params.result[result_index].x_re += 1.0f;
            params.result[result_index].y_re += 1.0f;
            params.result[result_index].z_re += 1.0f;
        }
    }
}

