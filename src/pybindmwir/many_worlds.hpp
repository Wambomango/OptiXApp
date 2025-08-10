#pragma once

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mwir/many_worlds.hpp"

class ManyWorlds
{

public:

    ManyWorlds();
    ManyWorlds(std::unique_ptr<MWIR::ManyWorlds> &&impl);
    ManyWorlds(torch::Tensor &min, torch::Tensor &max, float resolution, int n_samples);
    ManyWorlds Clone() const;

    void SetMin(torch::Tensor &min);
    void SetMax(torch::Tensor &max);
    void SetResolution(float resolution);
    void SetNSamples(int n_samples);
    torch::Tensor GetMin() const;
    torch::Tensor GetMax() const;
    float GetResolution() const;
    int GetNSamples() const;
    torch::Tensor GetOccupancy();
    torch::Tensor GetNormal();
    void UpdateNormal();


protected:
    friend class ManyWorldsRenderer;
    std::unique_ptr<MWIR::ManyWorlds> mwir_many_worlds_;
};


void init_many_worlds(pybind11::module_ &m);