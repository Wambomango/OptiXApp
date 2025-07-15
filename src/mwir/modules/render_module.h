#pragma once

#include "types.h"

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