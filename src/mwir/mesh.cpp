#include "mwir/include/mesh.hpp"
#include "mwir/mesh_impl.hpp"

namespace MWIR
{

Mesh::Mesh(std::string path)
{
    impl = new MeshImpl(path);
}

Mesh::Mesh(std::vector<glm::vec3> &&vertices)
{
    impl = new MeshImpl(std::move(vertices));
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
    if(impl)
    {
        impl->SetVertices(std::move(vertices));
    }
}

const std::vector<glm::vec3>& Mesh::GetVertices() const
{
    if(impl)
    {
        return impl->GetVertices();
    }
    
    static const std::vector<glm::vec3> emptyVertices;
    return emptyVertices;
}

}