#pragma once

#include "scene.hpp"
#include "window.hpp"
#include "camera.hpp"

#include <glad/glad.h>

#include <optix.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

class OptiXRenderer 
{
    public:
        OptiXRenderer(Window &window, Scene& scene);
        ~OptiXRenderer();

        void Render(Camera &camera);

    private:
        int width;
        int height;

        GLuint output_texture;
        cudaGraphicsResource* gl_cuda_resource;
        cudaArray* gl_cuda_array;

        CUdeviceptr image;
};