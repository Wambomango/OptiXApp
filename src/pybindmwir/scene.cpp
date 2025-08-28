#include "pybindmwir/scene.hpp"

namespace py = pybind11;

Scene::Scene()
{
    mwir_scene_ = std::make_unique<MWIR::Scene>(std::nullopt, std::nullopt, std::nullopt, std::nullopt);
}

Scene::Scene(std::unique_ptr<MWIR::Scene> &&impl)
{
    if (!impl)
    {
        throw std::invalid_argument("Scene implementation cannot be null.");
    }

    mwir_scene_ = std::move(impl);
}

Scene::Scene(std::shared_ptr<Mesh> mesh, std::vector<std::shared_ptr<Antenna>> senders, std::vector<std::shared_ptr<Antenna>> receivers, std::shared_ptr<Signal> signal)
{   
    std::vector<MWIR::Antenna> senders_mwir;
    for (auto& sender : senders)
    {
        senders_mwir.push_back(*(sender->mwir_antenna_));
    }
    std::vector<MWIR::Antenna> receivers_mwir;
    for (auto& receiver : receivers)
    {
        receivers_mwir.push_back(*(receiver->mwir_antenna_));
    }
    mwir_scene_ = std::make_unique<MWIR::Scene>(*(mesh->mwir_mesh_), senders_mwir, receivers_mwir, *(signal->mwir_signal_));
}

Scene Scene::Clone() const
{
    if (!mwir_scene_)
    {
        throw std::runtime_error("Scene ownership has been transferred.");
    }

    return Scene(std::move(std::make_unique<MWIR::Scene>(mwir_scene_->Clone())));
}

void Scene::SetMesh(std::shared_ptr<Mesh> &mesh)
{
    if (!mwir_scene_)
    {
        throw std::runtime_error("Scene ownership has been transferred");
    }

    mwir_scene_->SetMesh(*(mesh->mwir_mesh_));
}

void Scene::SetSenders(std::vector<std::shared_ptr<Antenna>> &senders)
{
    if (!mwir_scene_)
    {
        throw std::runtime_error("Scene ownership has been transferred");
    }

    std::vector<MWIR::Antenna> senders_mwir;
    for (auto& sender : senders)
    {
        senders_mwir.push_back(*(sender->mwir_antenna_));
    }
    mwir_scene_->SetSenders(senders_mwir);
}

void Scene::SetReceivers(std::vector<std::shared_ptr<Antenna>> &receivers)
{
    if (!mwir_scene_)
    {
        throw std::runtime_error("Scene ownership has been transferred");
    }

    std::vector<MWIR::Antenna> receivers_mwir;
    for (auto& receiver : receivers)
    {
        receivers_mwir.push_back(*(receiver->mwir_antenna_));
    }
    mwir_scene_->SetReceivers(receivers_mwir);
}

void Scene::SetSignal(std::shared_ptr<Signal> signal)
{
    if (!mwir_scene_)
    {
        throw std::runtime_error("Scene ownership has been transferred");
    }

    mwir_scene_->SetSignal(*(signal->mwir_signal_));
}

std::shared_ptr<Mesh> Scene::GetMesh()
{
    if (!mwir_scene_)
    {
        throw std::runtime_error("Scene ownership has been transferred");
    }

    std::shared_ptr<Mesh> tmp = std::make_shared<Mesh>(std::move(std::make_unique<MWIR::Mesh>(mwir_scene_->GetMesh())));
    return tmp;
}

std::vector<std::shared_ptr<Antenna>> Scene::GetSenders()
{
    if (!mwir_scene_)
    {
        throw std::runtime_error("Scene ownership has been transferred");
    }

    std::vector<MWIR::Antenna> senders_mwir = mwir_scene_->GetSenders();
    std::vector<std::shared_ptr<Antenna>> senders;
    for (auto& sender : senders_mwir)
    {
        senders.push_back(std::make_shared<Antenna>(std::move(std::make_unique<MWIR::Antenna>(sender))));
    }
    return senders;
}

std::vector<std::shared_ptr<Antenna>> Scene::GetReceivers()
{
    if (!mwir_scene_)
    {
        throw std::runtime_error("Scene ownership has been transferred");
    }

    std::vector<MWIR::Antenna> receivers_mwir = mwir_scene_->GetReceivers();
    std::vector<std::shared_ptr<Antenna>> receivers;
    for (auto& receiver : receivers_mwir)
    {
        receivers.push_back(std::make_shared<Antenna>(std::move(std::make_unique<MWIR::Antenna>(receiver))));
    }
    return receivers;
}

std::shared_ptr<Signal> Scene::GetSignal()
{
    if (!mwir_scene_)
    {
        throw std::runtime_error("Scene ownership has been transferred");
    }

    std::shared_ptr<Signal> tmp = std::make_shared<Signal>(std::move(std::make_unique<MWIR::Signal>(mwir_scene_->GetSignal())));
    return tmp;
}


void init_scene(py::module_ &m)
{
    py::class_<Scene, std::shared_ptr<Scene>>(m, "Scene")
        .def(py::init<>())
        .def(py::init<std::shared_ptr<Mesh>, std::vector<std::shared_ptr<Antenna>>, std::vector<std::shared_ptr<Antenna>>, std::shared_ptr<Signal>>())
        .def("Clone", &Scene::Clone)
        .def("SetMesh", &Scene::SetMesh)
        .def("SetSenders", &Scene::SetSenders)
        .def("SetReceivers", &Scene::SetReceivers)
        .def("SetSignal", &Scene::SetSignal)
        .def("GetMesh", &Scene::GetMesh)
        .def("GetSenders", &Scene::GetSenders)
        .def("GetReceivers", &Scene::GetReceivers)
        .def("GetSignal", &Scene::GetSignal);
}