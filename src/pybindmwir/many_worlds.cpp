#include "pybindmwir/many_worlds.hpp"

namespace py = pybind11;


ManyWorlds::ManyWorlds()
{ 
    mwir_many_worlds_ = std::make_unique<MWIR::ManyWorlds>(std::nullopt, std::nullopt, std::nullopt, std::nullopt);
}    

ManyWorlds::ManyWorlds(std::unique_ptr<MWIR::ManyWorlds> &&impl)
{
    if (!impl)
    {
        throw std::invalid_argument("ManyWorlds implementation cannot be null.");
    }

    mwir_many_worlds_ = std::move(impl);
}

ManyWorlds::ManyWorlds(torch::Tensor &min, torch::Tensor &max, float resolution, int n_samples)
{
    glm::vec3 min_vec = glm::vec3(min[0].item<float>(), min[1].item<float>(), min[2].item<float>());
    glm::vec3 max_vec = glm::vec3(max[0].item<float>(), max[1].item<float>(), max[2].item<float>());
    mwir_many_worlds_ = std::make_unique<MWIR::ManyWorlds>(min_vec, max_vec, resolution, n_samples);
}

ManyWorlds ManyWorlds::Clone() const
{
    if (!mwir_many_worlds_)
    {
        throw std::runtime_error("ManyWorlds ownership has been transferred.");
    }

    return ManyWorlds(std::move(std::make_unique<MWIR::ManyWorlds>(mwir_many_worlds_->Clone())));
}

void ManyWorlds::SetMin(torch::Tensor &min)
{
    if (!mwir_many_worlds_)
    {
        throw std::runtime_error("ManyWorlds ownership has been transferred.");
    }

    glm::vec3 min_vec = glm::vec3(min[0].item<float>(), min[1].item<float>(), min[2].item<float>());
    mwir_many_worlds_->SetMin(min_vec);
}

void ManyWorlds::SetMax(torch::Tensor &max)
{
    if (!mwir_many_worlds_)
    {
        throw std::runtime_error("ManyWorlds ownership has been transferred.");
    }

    glm::vec3 max_vec = glm::vec3(max[0].item<float>(), max[1].item<float>(), max[2].item<float>());
    mwir_many_worlds_->SetMax(max_vec);
}

void ManyWorlds::SetResolution(float resolution)
{
    if (!mwir_many_worlds_)
    {
        throw std::runtime_error("ManyWorlds ownership has been transferred.");
    }
    mwir_many_worlds_->SetResolution(resolution);
}

void ManyWorlds::SetNSamples(int n_samples)
{
    if (!mwir_many_worlds_)
    {
        throw std::runtime_error("ManyWorlds ownership has been transferred.");
    }
    mwir_many_worlds_->SetNSamples(n_samples);
}

torch::Tensor ManyWorlds::GetMin() const
{
    if (!mwir_many_worlds_)
    {
        throw std::runtime_error("ManyWorlds ownership has been transferred.");
    }

    glm::vec3 min = mwir_many_worlds_->GetMin();
    torch::Tensor result = torch::empty({3}, torch::kFloat32);
    result[0] = min.x;
    result[1] = min.y;
    result[2] = min.z;
    return result;
}

torch::Tensor ManyWorlds::GetMax() const
{
    if (!mwir_many_worlds_)
    {
        throw std::runtime_error("ManyWorlds ownership has been transferred.");
    }

    glm::vec3 max = mwir_many_worlds_->GetMax();
    torch::Tensor result = torch::empty({3}, torch::kFloat32);
    result[0] = max.x;
    result[1] = max.y;
    result[2] = max.z;
    return result;
}

float ManyWorlds::GetResolution() const
{
    if (!mwir_many_worlds_)
    {
        throw std::runtime_error("ManyWorlds ownership has been transferred.");
    }
    return mwir_many_worlds_->GetResolution();
}

int ManyWorlds::GetNSamples() const
{
    if (!mwir_many_worlds_)
    {
        throw std::runtime_error("ManyWorlds ownership has been transferred.");
    }
    return mwir_many_worlds_->GetNSamples();
}

torch::Tensor ManyWorlds::GetOccupancy()
{
    if (!mwir_many_worlds_)
    {
        throw std::runtime_error("ManyWorlds ownership has been transferred.");
    }
    return mwir_many_worlds_->GetOccupancy();
}


void init_many_worlds(py::module_ &m)
{
    py::class_<ManyWorlds, std::shared_ptr<ManyWorlds>>(m, "ManyWorlds")
        .def(py::init<>())
        .def(py::init<torch::Tensor&, torch::Tensor&, float, int>())
        .def("Clone", &ManyWorlds::Clone)
        .def("SetMin", &ManyWorlds::SetMin)
        .def("SetMax", &ManyWorlds::SetMax)
        .def("SetResolution", &ManyWorlds::SetResolution)
        .def("SetNSamples", &ManyWorlds::SetNSamples)
        .def("GetMin", &ManyWorlds::GetMin)
        .def("GetMax", &ManyWorlds::GetMax)
        .def("GetResolution", &ManyWorlds::GetResolution)
        .def("GetNSamples", &ManyWorlds::GetNSamples)
        .def("GetOccupancy", &ManyWorlds::GetOccupancy);
}