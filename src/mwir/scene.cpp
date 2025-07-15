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
        sender.impl = nullptr;
    }
    std::vector<AntennaImpl> receivers_impl;
    for (auto& receiver : receivers)
    {
        receivers_impl.push_back(std::move(*receiver.impl));
        receiver.impl = nullptr;
    }

    impl = new SceneImpl(std::move(*mesh.impl), std::move(senders_impl), std::move(receivers_impl), std::move(*signal.impl));
    mesh.impl = nullptr;
    signal.impl = nullptr;
}

Scene::~Scene()
{
    if (impl)
    {
        delete impl;
    }
}

Scene::Scene(Scene&& other) noexcept : impl(std::move(other.impl))
{
    other.impl = nullptr; 
}

Scene& Scene::operator=(Scene&& other) noexcept
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


void Scene::SetMesh(Mesh &&mesh)
{
    if (impl)
    {
        impl->SetMesh(std::move(*mesh.impl));
        mesh.impl = nullptr;
    }
}

void Scene::SetSenders(std::vector<Antenna> &&senders)
{
    std::vector<AntennaImpl> senders_impl;
    for (auto& sender : senders)
    {
        senders_impl.push_back(std::move(*sender.impl));
        sender.impl = nullptr;
    }

    if(impl)
    {
        impl->SetSenders(std::move(senders_impl));
    }
}

void Scene::SetReceivers(std::vector<Antenna> &&receivers)
{
    std::vector<AntennaImpl> receivers_impl;
    for (auto& receiver : receivers)
    {
        receivers_impl.push_back(std::move(*receiver.impl));
        receiver.impl = nullptr;
    }
    
    if(impl)
    {
        impl->SetReceivers(std::move(receivers_impl));
    }
}

void Scene::SetSignal(Signal &&signal)
{
    if (impl)
    {
        impl->SetSignal(std::move(*signal.impl));
        signal.impl = nullptr;
    }
}

}