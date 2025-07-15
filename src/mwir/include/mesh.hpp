#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <string>
namespace MWIR
{

class MeshImpl;

class Mesh
{

public:
    Mesh(std::string path);
    Mesh(std::vector<glm::vec3> &&vertices);
    ~Mesh();
    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;
    Mesh(Mesh&&) noexcept;
    Mesh& operator=(Mesh&&) noexcept;

    void SetVertices(std::vector<glm::vec3> &&vertices);
    const std::vector<glm::vec3>& GetVertices() const;

protected:
    friend class Scene;
    MeshImpl *impl;
};

}