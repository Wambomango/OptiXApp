#pragma once

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mwir/include/scene.hpp"

#include "mwir_python/mesh.hpp"
#include "mwir_python/antenna.hpp"
#include "mwir_python/signal.hpp"

namespace py = pybind11;

class Scene
{

public:
    Scene(std::shared_ptr<Mesh> mesh, std::vector<std::shared_ptr<Antenna>> senders, std::vector<std::shared_ptr<Antenna>> receivers, std::shared_ptr<Signal> signal)
    {

        // std::vector<MWIR::Antenna> senders_mwir;
        // for (auto& sender : senders)
        // {
        //     senders_mwir.push_back(std::move(*sender.mwir_antenna_));
        //     sender.mwir_antenna_.reset();
        // }
        // std::vector<MWIR::Antenna> receivers_mwir;
        // for (auto& receiver : receivers)
        // {
        //     receivers_mwir.push_back(std::move(*receiver.mwir_antenna_));
        //     receiver.mwir_antenna_.reset();
        // }

        // mwir_scene_ = std::make_unique<MWIR::Scene>(std::move(*mesh.mwir_mesh_), std::move(senders_mwir), std::move(receivers_mwir), std::move(*signal.mwir_signal_));
        // mesh.mwir_mesh_.reset();
        // signal.mwir_signal_.reset();
    }

    ~Scene()
    {
    }

    void SetMesh(Mesh &mesh)
    {
        // mwir_scene_->SetMesh(std::move(*mesh.mwir_mesh_));
        // mesh.mwir_mesh_.reset();
    }

    void SetSenders(std::vector<Antenna> &senders)
    {
        // std::vector<MWIR::Antenna> senders_mwir;
        // for (auto& sender : senders)
        // {
        //     senders_mwir.push_back(std::move(*sender.mwir_antenna_));
        //     sender.mwir_antenna_.reset();
        // }
        // mwir_scene_->SetSenders(std::move(senders_mwir));
    }

    void SetReceivers(std::vector<Antenna> &receivers)
    {
        // std::vector<MWIR::Antenna> receivers_mwir;
        // for (auto& receiver : receivers)
        // {
        //     receivers_mwir.push_back(std::move(*receiver.mwir_antenna_));
        //     receiver.mwir_antenna_.reset();
        // }
        // mwir_scene_->SetReceivers(std::move(receivers_mwir));
    }

    void SetSignal(Signal &signal)
    {
        // mwir_scene_->SetSignal(std::move(*signal.mwir_signal_));
        // signal.mwir_signal_.reset();
    }

private:
    std::unique_ptr<MWIR::Scene> mwir_scene_;
};

