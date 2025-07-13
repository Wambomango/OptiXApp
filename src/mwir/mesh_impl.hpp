#pragma once

#include <glm/glm.hpp>

#include <string>
#include <vector>

namespace MWIR
{

class MeshImpl
{
    public:
        MeshImpl(std::string path);
        MeshImpl(std::vector<glm::vec3> &&vertices);
        ~MeshImpl();
        MeshImpl(const MeshImpl&) = delete;
        MeshImpl& operator=(const MeshImpl&) = delete;
        MeshImpl(MeshImpl&&) noexcept;
        MeshImpl& operator=(MeshImpl&&) noexcept;

        void SetVertices(std::vector<glm::vec3> &&vertices);
        const std::vector<glm::vec3>& GetVertices() const;

    private:
        std::vector<glm::vec3> vertices;
};


}