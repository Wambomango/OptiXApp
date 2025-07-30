#include "mwir/include/mesh.hpp"
#include "mwir/mesh_impl.hpp"

#include <spdlog/spdlog.h>

namespace MWIR
{

Mesh::Mesh()
{
    impl = new MeshImpl();
}

Mesh::Mesh(std::vector<glm::vec3> &&vertices)
{
    impl = new MeshImpl(std::move(vertices));
}

Mesh::Mesh(MeshImpl &&mesh_impl)
{
    impl = new MeshImpl(std::move(mesh_impl));
}

Mesh::~Mesh()
{
    if(impl)
    {
        delete impl;
    }
}

Mesh::Mesh(Mesh&& other) noexcept : impl(std::move(other.impl))
{
    other.impl = nullptr; 
}

Mesh& Mesh::operator=(Mesh&& other) noexcept
{
    if (this != &other)
    {
        if(impl)
        {
            delete impl; 
        }
        impl = other.impl; 
        other.impl = nullptr; 
    }
    return *this;
}

void Mesh::SetVertices(std::vector<glm::vec3> &&vertices)
{
    if (!impl)
    {
        throw std::runtime_error("Mesh ownership has been moved");
    }

    impl->SetVertices(std::move(vertices));
}

const std::vector<glm::vec3>& Mesh::GetVertices() const
{
    if (!impl)
    {
        throw std::runtime_error("Mesh ownership has been moved");
    }

    return impl->GetVertices();
}

}