#pragma once

#include "complex.h"
#include "types.h"
#include "defines.h"
#include <curand_kernel.h>


struct SceneParams
{
    OptixTraversableHandle mesh_handle;
    unsigned int n_senders;
    AntennaData *h_senders;
    AntennaData *d_senders;
    unsigned int n_receivers;
    AntennaData *h_receivers;
    AntennaData *d_receivers;
    SignalData signal;
};


struct ManyWorldsParams
{
};


struct Params
{    
    SceneParams scene;
    ManyWorldsParams many_worlds;

    int antenna_index;
    complex3 *result;

    int seed;
    curandState *randstates;
};

struct RayGenData
{
};

struct MissData
{
};

struct HitData
{
};