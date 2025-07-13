#pragma once

#include <memory>
#include <glm/glm.hpp>
#include <vector>

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
    Mesh(Mesh&&) = default;
    Mesh& operator=(Mesh&&) = default;

    void SetVertices(std::vector<glm::vec3> &&vertices);
    const std::vector<glm::vec3>& GetVertices() const;

protected:
    friend class Scene;
    std::unique_ptr<MeshImpl> impl;
};

}