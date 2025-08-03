#pragma once

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mwir/signal.hpp"


class Signal
{

public:
    Signal();
    Signal(std::unique_ptr<MWIR::Signal> &&impl);
    Signal(const torch::Tensor &frequency_range, const torch::Tensor &n_samples);
    Signal Clone() const;

    void SetFrequencyRange(const torch::Tensor &frequency_range, const torch::Tensor &n_samples);
    torch::Tensor GetFrequencyRange() const;
    torch::Tensor GetNSamples() const;

    torch::Tensor GetFStep() const;

protected:
    friend class Scene;
    std::unique_ptr<MWIR::Signal> mwir_signal_;
};


void init_signal(pybind11::module_ &m);
