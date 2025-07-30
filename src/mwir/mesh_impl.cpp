#include "mwir/mesh_impl.hpp"
#include <spdlog/spdlog.h>

namespace MWIR
{


MeshImpl::MeshImpl()
{
    vertices.push_back(glm::vec3(123456.00f, 789101.00f, 112131.0f));
    vertices.push_back(glm::vec3(123456.00f, 789101.06f, 112131.0f));
    vertices.push_back(glm::vec3(123456.00f, 789101.06f, 112131.06f));
}

MeshImpl::MeshImpl(std::vector<glm::vec3> &&vertices) : vertices(std::move(vertices))
{
}

MeshImpl::~MeshImpl()
{
}

MeshImpl::MeshImpl(MeshImpl&& other) noexcept : vertices(std::move(other.vertices))
{
    other.vertices.clear();
}

MeshImpl& MeshImpl::operator=(MeshImpl&& other) noexcept
{
    if (this != &other)
    {
        vertices = std::move(other.vertices);
        other.vertices.clear();
    }
    
    return *this;
}

void MeshImpl::SetVertices(std::vector<glm::vec3> &&vertices)
{
    this->vertices = std::move(vertices);
}

const std::vector<glm::vec3> &MeshImpl::GetVertices() const
{
    return vertices;
}

}