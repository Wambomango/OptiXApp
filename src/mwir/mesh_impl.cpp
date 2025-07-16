#include "mwir/mesh_impl.hpp"

namespace MWIR
{

MeshImpl::MeshImpl(std::string path)
{
    // Load mesh from file
}

MeshImpl::MeshImpl(std::vector<glm::vec3> &&vertices) : vertices(std::move(vertices))
{
}

MeshImpl::~MeshImpl()
{
}

MeshImpl::MeshImpl(MeshImpl&& other) noexcept
{
    vertices = std::move(other.vertices);
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