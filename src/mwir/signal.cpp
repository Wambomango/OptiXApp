#include "mwir/signal.hpp"

#include <stdexcept>


namespace MWIR
{

Signal::Signal(std::optional<glm::vec2> frequency_range, std::optional<int> n_samples)
{
    data = std::make_shared<SignalData>();
    SetFrequencyRange(frequency_range, n_samples);
}

Signal Signal::Clone() const
{
    return Signal(data->frequency_range, data->n_samples);
}

void Signal::SetFrequencyRange(std::optional<glm::vec2> frequency_range, std::optional<int> n_samples)
{
    if(frequency_range.has_value()) 
    {
        if(frequency_range->x > frequency_range->y) 
        {
            throw std::invalid_argument("Frequency range start must be less than or equal to end.");
        }
        data->frequency_range = frequency_range.value();
    } 
    else 
    {
        data->frequency_range =  glm::vec2(1E9, 1E9);
    }
    data->frequency_range *= 2 * M_PI;

    if(n_samples.has_value()) 
    {
        if(n_samples.value() < 1) 
        {
            throw std::invalid_argument("Number of samples must be at least 1.");
        }
        data->n_samples = n_samples.value();
    } 
    else 
    {
        data->n_samples = 1;
    }

    if(data->n_samples == 1) 
    {
        if(data->frequency_range.x == data->frequency_range.y) 
        {
            data->f_step = 0.0f;
        } 
        else 
        {
            throw std::invalid_argument("Frequency range must match for a single frequency.");
        }
    } 
    else 
    {
        data->f_step = (data->frequency_range.y - data->frequency_range.x) / static_cast<float>(data->n_samples - 1);
    }
}

glm::vec2 Signal::GetFrequencyRange() const
{
    return data->frequency_range;
}

int Signal::GetNSamples() const 
{
    return data->n_samples;
}

float Signal::GetFStep() const 
{
    return data->f_step;
}


} 