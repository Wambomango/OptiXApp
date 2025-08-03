#include "pybindmwir/signal.hpp"

namespace py = pybind11;

Signal::Signal()
{
    mwir_signal_ = std::make_unique<MWIR::Signal>(std::nullopt, std::nullopt);
}

Signal::Signal(std::unique_ptr<MWIR::Signal> &&impl)
{
    if (!impl)
    {
        throw std::invalid_argument("Signal implementation cannot be null.");
    }

    mwir_signal_ = std::move(impl);
}

Signal::Signal(const torch::Tensor &frequency_range, const torch::Tensor &n_samples)
{
    mwir_signal_ = std::make_unique<MWIR::Signal>(glm::vec2(frequency_range[0].item<float>(), frequency_range[1].item<float>()), n_samples[0].item<int>());
}


Signal Signal::Clone() const
{
    if (!mwir_signal_)
    {
        throw std::runtime_error("Signal ownership has been transferred.");
    }

    return Signal(std::move(std::make_unique<MWIR::Signal>(mwir_signal_->Clone())));
}

void Signal::SetFrequencyRange(const torch::Tensor &frequency_range, const torch::Tensor &n_samples)
{
    if(!mwir_signal_)
    {
        throw std::runtime_error("Signal ownership has been transferred.");
    }

    mwir_signal_->SetFrequencyRange(glm::vec2(frequency_range[0].item<float>(), frequency_range[1].item<float>()), n_samples[0].item<int>());
}

torch::Tensor Signal::GetFrequencyRange() const
{
    if(!mwir_signal_)
    {
        throw std::runtime_error("Signal ownership has been transferred.");
    }

    glm::vec2 freq_range = mwir_signal_->GetFrequencyRange();
    return torch::tensor({freq_range.x, freq_range.y}, torch::kFloat32).view({2});
}

torch::Tensor Signal::GetNSamples() const
{
    if(!mwir_signal_)
    {
        throw std::runtime_error("Signal ownership has been transferred.");
    }

    return torch::tensor(mwir_signal_->GetNSamples(), torch::kInt32).view({1});
}

torch::Tensor Signal::GetFStep() const
{
    if(!mwir_signal_)
    {
        throw std::runtime_error("Signal ownership has been transferred.");
    }

    return torch::tensor(mwir_signal_->GetFStep(), torch::kFloat32).view({1});
}

void init_signal(py::module_ &m)
{
    py::class_<Signal, std::shared_ptr<Signal>>(m, "Signal")
        .def(py::init<>())
        .def(py::init<torch::Tensor&, torch::Tensor&>())
        .def("Clone", &Signal::Clone)
        .def("SetFrequencyRange", &Signal::SetFrequencyRange)
        .def("GetFrequencyRange", &Signal::GetFrequencyRange)
        .def("GetNSamples", &Signal::GetNSamples)
        .def("GetFStep", &Signal::GetFStep);
}