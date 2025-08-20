#pragma once

#include "mwir/modules/common.h"
#include "mwir/modules/defines.h"

__global__ void SetAntenna(Params *params, int antenna_index);
__global__ void MergeResults(Params *params);