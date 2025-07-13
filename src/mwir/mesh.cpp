#include "mwir/include/mesh.hpp"
#include "mwir/mesh_impl.hpp"

namespace MWIR
{

Mesh::Mesh(std::string path) : impl(std::make_unique<MeshImpl>(path))
{
}

Mesh::Mesh(std::vector<glm::vec3> &&vertices) : impl(std::make_unique<MeshImpl>(std::move(vertices)))
{
}

Mesh::~Mesh()
{
}

void Mesh::SetVertices(std::vector<glm::vec3> &&vertices)
{
    impl->SetVertices(std::move(vertices));
}

const std::vector<glm::vec3>& Mesh::GetVertices() const
{
    return impl->GetVertices();
}

}