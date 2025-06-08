#pragma once

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glad/glad.h>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ranges.h>


namespace OpenGL
{

inline void GLAPIENTRY glMessageCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar *message, const void *userParam)
{
    // ignore harmless error/warning codes
    if (id == 131169 || id == 131185 || id == 131218 || id == 131204)
    {
        return;
    }

    std::string source_string;
    switch (source)
    {
    case GL_DEBUG_SOURCE_API:
        source_string = "API";
        break;
    case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
        source_string = "Window System";
        break;
    case GL_DEBUG_SOURCE_SHADER_COMPILER:
        source_string = "Shader Compiler";
        break;
    case GL_DEBUG_SOURCE_THIRD_PARTY:
        source_string = "Third Party";
        break;
    case GL_DEBUG_SOURCE_APPLICATION:
        source_string = "Application";
        break;
    case GL_DEBUG_SOURCE_OTHER:
        source_string = "Other";
        break;
    }

    std::string type_string;
    switch (type)
    {
    case GL_DEBUG_TYPE_ERROR:
        type_string = "Error";
        break;
    case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
        type_string = "Deprecated Behaviour";
        break;
    case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
        type_string = "Undefined Behaviour";
        break;
    case GL_DEBUG_TYPE_PORTABILITY:
        type_string = "Portability";
        break;
    case GL_DEBUG_TYPE_PERFORMANCE:
        type_string = "Performance";
        break;
    case GL_DEBUG_TYPE_MARKER:
        type_string = "Marker";
        break;
    case GL_DEBUG_TYPE_PUSH_GROUP:
        type_string = "Push Group";
        break;
    case GL_DEBUG_TYPE_POP_GROUP:
        type_string = "Pop Group";
        break;
    case GL_DEBUG_TYPE_OTHER:
        type_string = "Other";
        break;
    }

    std::string severity_string;
    switch (severity)
    {
    case GL_DEBUG_SEVERITY_HIGH:
        severity_string = "high";
        break;
    case GL_DEBUG_SEVERITY_MEDIUM:
        severity_string = "medium";
        break;
    case GL_DEBUG_SEVERITY_LOW:
        severity_string = "low";
        break;
    case GL_DEBUG_SEVERITY_NOTIFICATION:
        severity_string = "notification";
        break;
    }

    if (type == GL_DEBUG_TYPE_ERROR)
    {
        SPDLOG_ERROR("[GL_ERROR] source:{} type:{} severity:{} message:{}", source_string, type_string, severity_string, message);
    }
    else
    {
        SPDLOG_DEBUG("[GL_DEBUG] source:{} type:{} severity:{} message:{}", source_string, type_string, severity_string, message);
    }
}

inline void glfwErrorCallback(int error, const char *description)
{
    SPDLOG_ERROR("[GLFW]: {}", description);
}

class Context
{
    public:
        virtual void Activate() = 0;
        virtual void Deactivate() = 0;
};

class ContextGLFW : public Context
{
    public:
        ContextGLFW(ContextGLFW *share = nullptr, bool debug = false, bool visible = false)
        {
            if (n_contexts == 0)
            {
                glfwSetErrorCallback(glfwErrorCallback);
                if (!glfwInit())
                {
                    SPDLOG_CRITICAL("glfw init failed");
                    exit(-1);
                }
            }

            if (share)
            {
                SPDLOG_DEBUG("creating shared context with {}", share->name);
            }

            glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
            glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
            glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
            glfwWindowHint(GLFW_VISIBLE, visible);
            glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, debug);
            glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);   
            window = glfwCreateWindow(1, 1, "opengl_context", NULL, share ? share->window : NULL);
            if (!window)
            {
                SPDLOG_CRITICAL("window creation failed");
                glfwTerminate();
                exit(-1);
            }
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

            glfwMakeContextCurrent(window);
            gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
            if (debug)
            {
                glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
                glEnable(GL_DEBUG_OUTPUT);
                glDebugMessageCallback(glMessageCallback, 0);
            }
            glfwMakeContextCurrent(NULL);

            n_contexts++;
            SPDLOG_DEBUG("initialized");
        }

        ~ContextGLFW()
        {
            glfwDestroyWindow(window);
            if (n_contexts == 1)
            {
                glfwTerminate();
            }
            n_contexts--;
        }

        void Activate() override
        {
            SPDLOG_DEBUG("activating");
            if (active)
            {
                SPDLOG_DEBUG("context already active");
                return;
            }
            active = true;
            glfwMakeContextCurrent(window);
        }

        void Deactivate() override
        {
            SPDLOG_DEBUG("deactivating");
            if (!active)
            {
                SPDLOG_DEBUG("context already inactive");
                return;
            }
            active = false;
            glfwMakeContextCurrent(NULL);
        }

        void Resize(int width, int height)
        {
            SPDLOG_DEBUG("resizing to {}x{}", width, height);
            glfwSetWindowSize(window, width, height);
        }

        GLFWwindow *window;

    private:
        bool active = false;
        inline static int n_contexts = 0;
        inline static int counter = 0;
};

}