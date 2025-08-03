
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



int main(int argc, char **argv)
{
    if (argc < 2)
    {
        SPDLOG_CRITICAL("No input string provided as the first argument.");
        return -1;
    }

    std::string filepath = argv[1];

    size_t width = 1024;
    size_t height = 1024;
    Window window(width, height, "MWIR Viewer");

    window.Render();
    Scene scene(filepath);
    window.PollEvents();
    GLRenderer gl_renderer(window, scene);
    window.PollEvents();
    OptiXRenderer optix_renderer(window, scene);
    window.PollEvents();

    Camera camera(90.0f, float(width) / float(height), 0.1f, 10000.0f);
    camera.AddCallbacks(window);
    
    bool use_optix = false;
    bool *use_optix_address = &use_optix;
    window.AddKeyCallback([use_optix_address]
    (int key, int scancode, int action, int mods) 
    {
        if (key == GLFW_KEY_R && action == GLFW_PRESS)
        {
            *use_optix_address = !*use_optix_address;
        }
    });


    float last_time = window.GetTime();
    while (!window.ShouldClose())
    {
        float time = window.GetTime();
        float dt = time - last_time;
        last_time = time;

        camera.Tick(dt);

        if (use_optix)
        {
            optix_renderer.Render(camera);
        }
        else
        {
            gl_renderer.Render(camera);
        }

        window.Render();
        window.SwapBuffers();
        window.PollEvents();
    }

    return 0;
}