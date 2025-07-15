#pragma once

#include <glm/glm.hpp>
#include <vector>

namespace MWIR
{

class SignalImpl;

class Signal
{

public:
    Signal(glm::vec2 frequency_range, int n_samples);
    ~Signal();
    Signal(const Signal&) = delete;
    Signal& operator=(const Signal&) = delete;
    Signal(Signal&&) noexcept;
    Signal& operator=(Signal&&) noexcept;

    glm::vec2 GetFrequencyRange() const;
    int GetNFrequencies() const;
    float GetFStep() const;

protected:
    friend class Scene;
    SignalImpl *impl;
};

}