#pragma once

#include <glm/glm.hpp>

namespace MWIR
{

class SignalImpl
{
public:
    SignalImpl();
    SignalImpl(glm::vec2 frequency_range, int n_samples);
    ~SignalImpl();
    SignalImpl(const SignalImpl&) = delete;
    SignalImpl& operator=(const SignalImpl&) = delete;
    SignalImpl(SignalImpl&&) = default;
    SignalImpl& operator=(SignalImpl&&) = default;

    void SetFrequencyRange(glm::vec2 frequency_range, int n_samples);
    glm::vec2 GetFrequencyRange() const;
    int GetNSamples() const;
    float GetFStep() const;

private:
    glm::vec2 frequency_range;
    int n_samples;
    float f_step;
}; 

}