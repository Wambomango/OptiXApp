#pragma once

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mwir/scene.hpp"

#include "pybindmwir/mesh.hpp"
#include "pybindmwir/antenna.hpp"
#include "pybindmwir/signal.hpp"


class Scene
{

public:
    Scene();
    Scene(std::unique_ptr<MWIR::Scene> &&impl);
    Scene(std::shared_ptr<Mesh> mesh, std::vector<std::shared_ptr<Antenna>> senders, std::vector<std::shared_ptr<Antenna>> receivers, std::shared_ptr<Signal> signal);
    Scene Clone() const;

    void SetMesh(std::shared_ptr<Mesh> &mesh);
    void SetSenders(std::vector<std::shared_ptr<Antenna>> &senders);
    void SetReceivers(std::vector<std::shared_ptr<Antenna>> &receivers);
    void SetSignal(std::shared_ptr<Signal> signal);
    std::shared_ptr<Mesh> GetMesh();
    std::vector<std::shared_ptr<Antenna>> GetSenders();
    std::vector<std::shared_ptr<Antenna>> GetReceivers();
    std::shared_ptr<Signal> GetSignal();

protected:
    friend class Renderer;
    friend class ManyWorldsRenderer;
    std::unique_ptr<MWIR::Scene> mwir_scene_;
};


void init_scene(pybind11::module_ &);
