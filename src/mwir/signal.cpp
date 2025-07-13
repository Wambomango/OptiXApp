#include "mwir/include/signal.hpp"
#include "mwir/signal_impl.hpp"

namespace MWIR
{

Signal::Signal(glm::vec2 frequency_range, int n_samples) : impl(std::make_unique<SignalImpl>(frequency_range, n_samples))
{
}

Signal::~Signal()
{
}

glm::vec2 Signal::GetFrequencyRange() const
{
    return impl->GetFrequencyRange();
}

int Signal::GetNFrequencies() const
{
    return impl->GetNFrequencies();
}

float Signal::GetFStep() const
{
    return impl->GetFStep();
}

}