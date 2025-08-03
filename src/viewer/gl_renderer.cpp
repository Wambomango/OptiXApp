#include "gl_renderer.hpp"

#include "utils/opengl/texture.hpp"
#include "bindings.hpp"

#include <oneapi/tbb.h>

GLRenderer::GLRenderer(Window &window, Scene& scene) : ssao(window.GetWidth(), window.GetHeight())
{
    width = window.GetWidth();
    height = window.GetHeight();
    output_texture = window.GetTexture();

    GL_CREATE_TEXTURE_2D(g_position_texture, GL_RGBA32F, width, height);
    glBindTextureUnit(TEXTURE_UNIT_POSITION, g_position_texture);

    GL_CREATE_TEXTURE_2D(g_normal_texture, GL_RGB32F, width, height);
    glBindTextureUnit(TEXTURE_UNIT_NORMAL, g_normal_texture);

    GL_CREATE_TEXTURE_2D(g_depth_texture, GL_DEPTH_COMPONENT32F, width, height);
    glBindTextureUnit(TEXTURE_UNIT_DEPTH, g_depth_texture);

    glCreateFramebuffers(1, &g_framebuffer);
    glNamedFramebufferTexture(g_framebuffer, GL_COLOR_ATTACHMENT0, g_position_texture, 0);
    glNamedFramebufferTexture(g_framebuffer, GL_COLOR_ATTACHMENT1, g_normal_texture, 0);
    glNamedFramebufferTexture(g_framebuffer, GL_DEPTH_ATTACHMENT, g_depth_texture, 0);
    glCheckNamedFramebufferStatus(g_framebuffer, GL_FRAMEBUFFER);

    GLenum draw_buffers[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
    glNamedFramebufferDrawBuffers(g_framebuffer, 2, draw_buffers);


    glCreateFramebuffers(1, &deferred_framebuffer);
    glNamedFramebufferTexture(deferred_framebuffer, GL_COLOR_ATTACHMENT0, window.GetTexture(), 0);
    glCheckNamedFramebufferStatus(deferred_framebuffer, GL_FRAMEBUFFER);

    scene_buffer = std::make_unique<OpenGL::Buffer>(2 * 1000 * 1000 * 1000, GL_STATIC_DRAW);
    prepass_vao.SetVertexBufferAndLayout(*scene_buffer, {{GL_FLOAT, 3, false}, {GL_FLOAT, 3, false}});    
    prepass_program = std::make_unique<OpenGL::Program>(SHADER_DIR + std::string("prepass.vert"), SHADER_DIR + std::string("prepass.frag"));

    deferred_program = std::make_unique<OpenGL::Program>(SHADER_DIR + std::string("deferred.vert"), SHADER_DIR + std::string("deferred.frag"));

    torch::Tensor vertices, indices;
    scene.GetMesh(vertices, indices);

    n_vertices = indices.size(0) * 3;
    std::vector<std::pair<glm::vec3, glm::vec3>> scene_buffer_cpu(n_vertices);
    glm::vec3 *vertex_data = (glm::vec3 *)vertices.data_ptr();
    uint32_t *index_data = (uint32_t *)indices.data_ptr();

    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, indices.size(0)),
    [&](const oneapi::tbb::blocked_range<size_t> &r) {
        for (size_t i = r.begin(); i < r.end(); i++)
        {
            glm::vec3 v0 = vertex_data[index_data[i * 3 + 0]];
            glm::vec3 v1 = vertex_data[index_data[i * 3 + 1]];
            glm::vec3 v2 = vertex_data[index_data[i * 3 + 2]];
            glm::vec3 normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));
            scene_buffer_cpu[i * 3 + 0] = {v0, normal};
            scene_buffer_cpu[i * 3 + 1] = {v1, normal};
            scene_buffer_cpu[i * 3 + 2] = {v2, normal};
        }
    });


    // for(int i = 0; i < indices.size(0); i++)
    // {
    //     glm::vec3 v0 = vertex_data[index_data[i * 3 + 0]];
    //     glm::vec3 v1 = vertex_data[index_data[i * 3 + 1]];
    //     glm::vec3 v2 = vertex_data[index_data[i * 3 + 2]];
    //     glm::vec3 normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));
    //     scene_buffer_cpu[i * 3 + 0] = {v0, normal};
    //     scene_buffer_cpu[i * 3 + 1] = {v1, normal};
    //     scene_buffer_cpu[i * 3 + 2] = {v2, normal};
    // }

    scene_buffer->Store(scene_buffer_cpu.data(), n_vertices * sizeof(std::pair<glm::vec3, glm::vec3>), 0);
}

GLRenderer::~GLRenderer() 
{
    prepass_program.release();
    deferred_program.release();
    glDeleteTextures(1, &g_position_texture);
    glDeleteTextures(1, &g_normal_texture);
    glDeleteTextures(1, &g_depth_texture);
    glDeleteFramebuffers(1, &g_framebuffer);
    glDeleteFramebuffers(1, &deferred_framebuffer);
}

void GLRenderer::Render(Camera &camera) 
{
    glm::vec3 light_direction = glm::normalize(glm::vec3(0.0, -1.0, 0.0));
    glm::vec3 light_color = glm::vec3(0.8f, 0.8f, 0.8f);

    glEnable(GL_DEPTH_TEST);
    glBindFramebuffer(GL_FRAMEBUFFER, g_framebuffer);
    glViewport(0, 0, width, height);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    prepass_vao.Bind();
    prepass_program->SetMat4("view", 1, camera.GetViewMatrix());
    prepass_program->SetMat4("projection", 1, camera.GetProjectionMatrix());
    prepass_program->DrawArrays(GL_TRIANGLES, 0, n_vertices);

    ssao.CalculateSSAO(camera);    

    glBindFramebuffer(GL_FRAMEBUFFER, deferred_framebuffer);
    glViewport(0, 0, width, height);
    glClearColor(pow(0.5f, 1.0f / 2.2f), pow(0.6f, 1.0f / 2.2f), pow(0.9f, 1.0f / 2.2f), 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    deferred_vao.Bind();
    deferred_program->SetVec3("light_direction", 1, light_direction);
    deferred_program->SetVec3("light_color", 1, light_color);
    deferred_program->SetVec3("camera_position", 1, camera.GetPosition());
    deferred_program->DrawArrays(GL_TRIANGLES, 0, 3);
}