
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



int main(int arg, char **argv)
{
    if (!glfwInit())
    {
        SPDLOG_CRITICAL("glfw init failed");
        exit(-1);
    }

    size_t width = 1024;
    size_t height = 1024;
    Window window(width, height, "OptiX App");

    Scene scene("/home/mario/Desktop/Masterarbeit/OptiXApp/scenes/trees/Tree1.obj");
    GLRenderer gl_renderer(window, scene);
    OptiXRenderer optix_renderer(window, scene);

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