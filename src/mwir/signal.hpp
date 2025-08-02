#pragma once

#include <glm/glm.hpp>
#include <optional>
#include <memory>

namespace MWIR
{

class Signal
{
public:
    Signal(std::optional<glm::vec2> frequency_range, std::optional<int> n_samples);
    Signal Clone() const;

    void SetFrequencyRange(std::optional<glm::vec2> frequency_range, std::optional<int> n_samples);
    glm::vec2 GetFrequencyRange() const;
    int GetNSamples() const;
    float GetFStep() const;

private:
    struct SignalData
    {
        glm::vec2 frequency_range;
        int n_samples;
        float f_step;
    };

    std::shared_ptr<SignalData> data;
}; 

}