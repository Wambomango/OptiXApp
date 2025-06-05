
#include "utils/optix/context.hpp"
#include "utils/optix/module.hpp"
#include "utils/optix/program_group.hpp"
#include "utils/optix/pipeline.hpp"
#include "utils/optix/sbt_record.hpp"

#include "window.hpp"
#include "camera.hpp"
#include "scene.hpp"
#include "gl_renderer.hpp"
#include "optix_renderer.hpp"

#include <glm/gtx/string_cast.hpp>

int main(int arg, char **argv)
{
    size_t width = 1024;
    size_t height = 1024;
    Window window(width, height, "OptiX App");
    Camera camera(45.0f, float(width) / float(height), 0.1f, 100.0f);
    camera.SetCallbacks(window);

    Scene scene("/home/mario/Desktop/Masterarbeit/OptiXApp/scenes/trees/Tree1.obj");

    GLRenderer gl_renderer(window, scene);
    OptiXRenderer optix_renderer(window, scene);

    while (!window.ShouldClose())
    {
        gl_renderer.Render(camera);
        window.Render();

        window.SwapBuffers();
        window.PollEvents();
    }

    return 0;
}