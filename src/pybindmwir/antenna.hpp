#pragma once

#include <torch/torch.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mwir/antenna.hpp"

class Antenna
{

public:

    Antenna();
    Antenna(torch::Tensor &position, torch::Tensor &euler, torch::Tensor &fov, torch::Tensor &ray_density);
    Antenna(std::unique_ptr<MWIR::Antenna> &&impl);
    Antenna Clone() const;

    void SetPosition(const torch::Tensor &position);
    void SetOrientation(const torch::Tensor &euler);
    void SetFOV(const torch::Tensor &fov);
    void SetRayDensity(const torch::Tensor &ray_density);
    torch::Tensor GetPosition() const;
    torch::Tensor GetOrientation() const;
    torch::Tensor GetFOV() const;
    torch::Tensor GetRayDensity() const;
    torch::Tensor GetSolidAngle() const;    
    torch::Tensor GetNRays() const; 
  

protected:
    friend class Scene;
    std::unique_ptr<MWIR::Antenna> mwir_antenna_;
};


void init_antenna(pybind11::module_ &m);