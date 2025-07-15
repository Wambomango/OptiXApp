#pragma once


#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mwir/include/signal.hpp"

namespace py = pybind11;

class Signal
{

public:
    Signal(const torch::Tensor &frequency_range, const torch::Tensor &n_frequencies)
    {
        if (frequency_range.sizes() != std::vector<int64_t>{2})
        {
            throw std::invalid_argument("frequency_range must be a tensor of shape [2]");
        }

        if (n_frequencies.sizes() != std::vector<int64_t>{1})
        {
            throw std::invalid_argument("n_frequencies must be a tensor of shape [1]");
        }

        mwir_signal_ = std::make_unique<MWIR::Signal>(glm::vec2(frequency_range[0].item<float>(), frequency_range[1].item<float>()), n_frequencies[0].item<int>());
    }

    Signal(py::tuple frequency_range, int n_frequencies)
    {
        if (frequency_range.size() != 2)
        {
            throw std::invalid_argument("frequency_range must be a tuple of length 2");
        }

        mwir_signal_ = std::make_unique<MWIR::Signal>(glm::vec2(frequency_range[0].cast<float>(), frequency_range[1].cast<float>()), n_frequencies);
    }

    ~Signal()
    {
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

    torch::Tensor GetNFrequencies() const
    {
        if(!mwir_signal_)
        {
            throw std::runtime_error("Signal ownership has been transferred.");
        }

        return torch::tensor(mwir_signal_->GetNFrequencies(), torch::kInt32).view({1});
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

