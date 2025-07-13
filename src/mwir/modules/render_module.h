#pragma once

#include "types.h"

struct Params
{    
    OptixTraversableHandle mesh_handle;
    unsigned int n_senders;
    AntennaData *senders;
    unsigned int n_receivers;
    AntennaData *receivers;
    SignalData signal;

    int batch_number;
    EField *result;
};

struct RayGenData
{
    unsigned int color;
};

struct MissData
{
};

struct HitData
{
};