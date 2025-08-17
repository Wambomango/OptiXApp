#pragma once

#include "mwir/modules/render_module.h"
#include "mwir/modules/complex.h"

__global__ void SetAntenna(Params *params, int antenna_index);
__global__ void MergeResults(Params *params);