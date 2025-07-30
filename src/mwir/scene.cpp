#include "mwir/include/scene.hpp"
#include "mwir/mesh_impl.hpp"
#include "mwir/antenna_impl.hpp"
#include "mwir/signal_impl.hpp"
#include "mwir/scene_impl.hpp"

namespace MWIR
{

Scene::Scene()
{
    MeshImpl dummy_mesh;
    std::vector<AntennaImpl> dummy_senders(1);
    std::vector<AntennaImpl> dummy_receivers(1);
    SignalImpl dummy_signal;
    impl = new SceneImpl(std::move(dummy_mesh), std::move(dummy_senders), std::move(dummy_receivers), std::move(dummy_signal));
}

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

Scene::Scene(SceneImpl &&scene_impl)
{
    impl = new SceneImpl(std::move(scene_impl));
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
    if (!impl)
    {
        throw std::runtime_error("Scene ownership has been transferred");
    }

    impl->SetMesh(std::move(*mesh.impl));
    mesh.impl = nullptr;
}

void Scene::SetSenders(std::vector<Antenna> &&senders)
{
    if (!impl)
    {
        throw std::runtime_error("Scene ownership has been transferred");
    }

    std::vector<AntennaImpl> senders_impl;
    for (auto& sender : senders)
    {
        senders_impl.push_back(std::move(*sender.impl));
        sender.impl = nullptr;
    }
    impl->SetSenders(std::move(senders_impl));
}

void Scene::SetReceivers(std::vector<Antenna> &&receivers)
{
    if (!impl)
    {
        throw std::runtime_error("Scene ownership has been transferred");
    }

    std::vector<AntennaImpl> receivers_impl;
    for (auto& receiver : receivers)
    {
        receivers_impl.push_back(std::move(*receiver.impl));
        receiver.impl = nullptr;
    }
    impl->SetReceivers(std::move(receivers_impl));
}

void Scene::SetSignal(Signal &&signal)
{
    if (!impl)
    {
        throw std::runtime_error("Scene ownership has been transferred");
    }

    impl->SetSignal(std::move(*signal.impl));
    signal.impl = nullptr;
}

Mesh Scene::GetMesh()
{
    if(!impl)
    {
        throw std::runtime_error("Scene ownership has been transferred");
    }

    Mesh tmp(impl->GetMesh());
    return tmp;
}

std::vector<Antenna> Scene::GetSenders()
{
    if (!impl)
    {
        throw std::runtime_error("Scene ownership has been transferred");
    }

    std::vector<AntennaImpl> tmp = impl->GetSenders();
    std::vector<Antenna> result;
    result.reserve(tmp.size());
    for (auto& sender_impl : tmp)
    {
        result.emplace_back(std::move(sender_impl));
    }
    return result;
}

std::vector<Antenna> Scene::GetReceivers()
{
    if (!impl)
    {
        throw std::runtime_error("Scene ownership has been transferred");
    }

    std::vector<AntennaImpl> tmp = impl->GetReceivers();
    std::vector<Antenna> result;
    result.reserve(tmp.size());
    for (auto& receiver_impl : tmp)
    {
        result.emplace_back(std::move(receiver_impl));
    }
    return result;
}

Signal Scene::GetSignal()
{
    if (!impl)
    {
        throw std::runtime_error("Scene ownership has been transferred");
    }

    Signal tmp(impl->GetSignal());
    return tmp;
}

}