#pragma once

#include "types.h"

#define OPTIX_MAX_GRID_DIM 256
struct Params
{    
    OptixTraversableHandle mesh_handle;
    unsigned int n_senders;
    AntennaData *h_senders;
    AntennaData *d_senders;
    unsigned int n_receivers;
    AntennaData *h_receivers;
    AntennaData *d_receivers;
    SignalData signal;

    int antenna_index;
    int grid_x;
    int grid_y;
    EField *result;
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