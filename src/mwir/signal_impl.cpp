#include "mwir/signal_impl.hpp"

#include <stdexcept>


namespace MWIR
{

SignalImpl::SignalImpl()
{
    frequency_range = glm::vec2(2 * M_PI * 1E9, 2 * M_PI * 1E9);
    n_samples = 1;
    SetFrequencyRange(frequency_range, n_samples);
}

SignalImpl::SignalImpl(glm::vec2 frequency_range, int n_samples)
{
    SetFrequencyRange(frequency_range, n_samples);
}

SignalImpl::~SignalImpl()
{
}

void SignalImpl::SetFrequencyRange(glm::vec2 frequency_range, int n_samples)
{
    this->frequency_range = frequency_range;
    this->n_samples = n_samples;

    if(n_samples == 1) 
    {
        if(frequency_range.x == frequency_range.y) 
        {
            f_step = 0.0f;
        } 
        else 
        {
            throw std::invalid_argument("Frequency range must match for a single frequency.");
        }
    } 
    else 
    {
        f_step = (frequency_range.y - frequency_range.x) / static_cast<float>(n_samples - 1);
    }
}


glm::vec2 SignalImpl::GetFrequencyRange() const {
    return frequency_range;
}

int SignalImpl::GetNSamples() const {
    return n_samples;
}

float SignalImpl::GetFStep() const {
    return f_step;
}
} 