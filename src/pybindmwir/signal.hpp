#pragma once


#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mwir/include/signal.hpp"

namespace py = pybind11;

class Signal
{

public:
    Signal()
    {
        mwir_signal_ = std::make_unique<MWIR::Signal>();
    }

    Signal(MWIR::Signal &&mwir_signal)
    {
        mwir_signal_ = std::make_unique<MWIR::Signal>(std::move(mwir_signal));
    }

    Signal(const torch::Tensor &frequency_range, const torch::Tensor &n_samples)
    {
        mwir_signal_ = std::make_unique<MWIR::Signal>(glm::vec2(frequency_range[0].item<float>(), frequency_range[1].item<float>()), n_samples[0].item<int>());
    }

    ~Signal()
    {
    }


    void SetFrequencyRange(const torch::Tensor &frequency_range, const torch::Tensor &n_samples)
    {
        if(!mwir_signal_)
        {
            throw std::runtime_error("Signal ownership has been transferred.");
        }

        mwir_signal_->SetFrequencyRange(glm::vec2(frequency_range[0].item<float>(), frequency_range[1].item<float>()), n_samples[0].item<int>());
    }

    torch::Tensor GetFrequencyRange() const
    {
        if(!mwir_signal_)
        {
            throw std::runtime_error("Signal ownership has been transferred.");
        }

        glm::vec2 freq_range = mwir_signal_->GetFrequencyRange();
        return torch::tensor({freq_range.x, freq_range.y}, torch::kFloat32).view({2});
    }

    torch::Tensor GetNSamples() const
    {
        if(!mwir_signal_)
        {
            throw std::runtime_error("Signal ownership has been transferred.");
        }

        return torch::tensor(mwir_signal_->GetNSamples(), torch::kInt32).view({1});
    }

    torch::Tensor GetFStep() const
    {
        if(!mwir_signal_)
        {
            throw std::runtime_error("Signal ownership has been transferred.");
        }

        return torch::tensor(mwir_signal_->GetFStep(), torch::kFloat32).view({1});
    }


protected:
    friend class Scene;
    std::unique_ptr<MWIR::Signal> mwir_signal_;
};

