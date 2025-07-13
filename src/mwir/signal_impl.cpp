#include "mwir/signal_impl.hpp"

namespace MWIR
{
SignalImpl::SignalImpl(glm::vec2 frequency_range, int n_frequencies) : frequency_range(frequency_range), n_frequencies(n_frequencies) 
{
    f_step = (frequency_range.y - frequency_range.x) / static_cast<float>(n_frequencies - 1);
}

SignalImpl::~SignalImpl()
{
}

glm::vec2 SignalImpl::GetFrequencyRange() const {
    return frequency_range;
}

int SignalImpl::GetNFrequencies() const {
    return n_frequencies;
}

float SignalImpl::GetFStep() const {
    return f_step;
}
} 