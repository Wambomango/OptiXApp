#include "mwir/include/signal.hpp"
#include "mwir/signal_impl.hpp"

namespace MWIR
{

Signal::Signal(glm::vec2 frequency_range, int n_samples)
{
    impl = new SignalImpl(frequency_range, n_samples);

}

Signal::~Signal()
{
    if (impl)
    {
        delete impl;
    }
}

Signal::Signal(Signal&& other) noexcept : impl(std::move(other.impl))
{
    other.impl = nullptr; 
} 

Signal& Signal::operator=(Signal&& other) noexcept
{
    if (this != &other)
    {
        if (impl)
        {
            delete impl; 
        }
        impl = other.impl; 
        other.impl = nullptr; 
    }
    return *this;
}

glm::vec2 Signal::GetFrequencyRange() const
{
    if (impl)
    {
        return impl->GetFrequencyRange();
    }

    return glm::vec2(0.0f, 0.0f);
}

int Signal::GetNFrequencies() const
{
    if (impl)
    {
        return impl->GetNFrequencies();
    }
    return 0;
}

float Signal::GetFStep() const
{
    if (impl)
    {
        return impl->GetFStep();
    }
    return 0.0f;
}

}