#include "mwir/include/scene.hpp"
#include "mwir/mesh_impl.hpp"
#include "mwir/antenna_impl.hpp"
#include "mwir/signal_impl.hpp"
#include "mwir/scene_impl.hpp"

namespace MWIR
{

Scene::Scene(Mesh &&mesh, std::vector<Antenna> &&senders, std::vector<Antenna> &&receivers, Signal &&signal) 
{
    std::vector<AntennaImpl> senders_impl;
    for (auto& sender : senders)
    {
        senders_impl.push_back(std::move(*sender.impl));
        sender.impl.reset();
    }
    std::vector<AntennaImpl> receivers_impl;
    for (auto& receiver : receivers)
    {
        receivers_impl.push_back(std::move(*receiver.impl));
        receiver.impl.reset();
    }

    impl = std::make_unique<SceneImpl>(std::move(*mesh.impl), std::move(senders_impl), std::move(receivers_impl), std::move(*signal.impl));
    mesh.impl.reset();
    signal.impl.reset();
}

Scene::~Scene()
{
}

void Scene::SetMesh(Mesh &&mesh)
{
    impl->SetMesh(std::move(*mesh.impl));
    mesh.impl.reset();
}

void Scene::SetSenders(std::vector<Antenna> &&senders)
{
    std::vector<AntennaImpl> senders_impl;
    for (auto& sender : senders)
    {
        senders_impl.push_back(std::move(*sender.impl));
        sender.impl.reset();
    }
    impl->SetSenders(std::move(senders_impl));
}

void Scene::SetReceivers(std::vector<Antenna> &&receivers)
{
    std::vector<AntennaImpl> receivers_impl;
    for (auto& receiver : receivers)
    {
        receivers_impl.push_back(std::move(*receiver.impl));
        receiver.impl.reset();
    }
    impl->SetReceivers(std::move(receivers_impl));
}

void Scene::SetSignal(Signal &&signal)
{
    SignalImpl signal_impl = std::move(*signal.impl);
    signal.impl.reset();
}

}