#pragma once

#include "mesh.hpp"
#include "antenna.hpp"
#include "signal.hpp"

#include <glm/glm.hpp>
#include <vector>

namespace MWIR
{

class SceneImpl;

class Scene
{

public:
    Scene();
    Scene(Mesh &&mesh, std::vector<Antenna> &&senders, std::vector<Antenna> &&receivers, Signal &&signal);
    Scene(SceneImpl &&scene_impl);
    ~Scene();
    Scene(const Scene&) = delete;
    Scene& operator=(const Scene&) = delete;
    Scene(Scene&&) noexcept;
    Scene& operator=(Scene&&) noexcept;

    void SetMesh(Mesh &&mesh);
    void SetSenders(std::vector<Antenna> &&senders);
    void SetReceivers(std::vector<Antenna> &&receivers);
    void SetSignal(Signal &&signal);

    Mesh GetMesh();
    std::vector<Antenna> GetSenders();
    std::vector<Antenna> GetReceivers();
    Signal GetSignal();

protected:
    friend class Renderer;
    SceneImpl *impl;

};

}