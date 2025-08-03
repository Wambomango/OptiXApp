
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pybindmwir/antenna.hpp"
#include "pybindmwir/many_worlds.hpp"
#include "pybindmwir/mesh.hpp"
#include "pybindmwir/forward_renderer.hpp"
#include "pybindmwir/inverse_renderer.hpp"
#include "pybindmwir/scene.hpp"
#include "pybindmwir/signal.hpp"

PYBIND11_MODULE(PyBindMWIR, m) 
{
    m.doc() = "PyBindMWIR python module";
    init_antenna(m);
    init_many_worlds(m);
    init_mesh(m);
    init_forward_renderer(m);
    init_inverse_renderer(m);
    init_scene(m);
    init_signal(m);
}





