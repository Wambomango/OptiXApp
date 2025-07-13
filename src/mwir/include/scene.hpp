#pragma once

#include "mesh.hpp"
#include "antenna.hpp"
#include "signal.hpp"

#include <memory>
#include <glm/glm.hpp>
#include <vector>

namespace MWIR
{

class SceneImpl;

class Scene
{

public:

    Scene(Mesh &&mesh, std::vector<Antenna> &&senders, std::vector<Antenna> &&receivers, Signal &&signal);
    ~Scene();
    Scene(const Scene&) = delete;
    Scene& operator=(const Scene&) = delete;
    Scene(Scene&&) = default;
    Scene& operator=(Scene&&) = default;

    void SetMesh(Mesh &&mesh);
    void SetSenders(std::vector<Antenna> &&senders);
    void SetReceivers(std::vector<Antenna> &&receivers);
    void SetSignal(Signal &&signal);


protected:
    friend class Renderer;
    std::unique_ptr<SceneImpl> impl;
};

}