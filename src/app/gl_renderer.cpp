#include "gl_renderer.hpp"

#include "utils/opengl/texture.hpp"
#include "bindings.hpp"


GLRenderer::GLRenderer(Window &window, Scene& scene) 
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

    scene_buffer = std::make_unique<OpenGL::Buffer>(1000 * 1000 * 1000, GL_STATIC_DRAW);
    prepass_vao.SetVertexBufferAndLayout(*scene_buffer, {{GL_FLOAT, 3, false}, {GL_FLOAT, 3, false}});    
    prepass_program = std::make_unique<OpenGL::Program>(SHADER_DIR + std::string("prepass.vert"), SHADER_DIR + std::string("prepass.frag"));

    deferred_program = std::make_unique<OpenGL::Program>(SHADER_DIR + std::string("deferred.vert"), SHADER_DIR + std::string("deferred.frag"));

    auto& attrib = scene.GetAttrib();
    auto& shapes = scene.GetShapes();
    auto& materials = scene.GetMaterials();

    std::vector<Scene::SceneVertex> scene_buffer_cpu;
    n_vertices = 0;
    for (size_t s = 0; s < shapes.size(); s++)
    {
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) 
        {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

            for (size_t v = 0; v < fv; v++) 
            {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                tinyobj::real_t vx = attrib.vertices[3*size_t(idx.vertex_index)+0];
                tinyobj::real_t vy = attrib.vertices[3*size_t(idx.vertex_index)+1];
                tinyobj::real_t vz = attrib.vertices[3*size_t(idx.vertex_index)+2];
                scene_buffer_cpu.push_back(Scene::SceneVertex{
                    glm::vec3(vx, vy, vz), 
                    glm::vec3(0, 0, 0)
                });

                n_vertices++;
            }

            glm::vec3 normal = glm::cross(
                scene_buffer_cpu[n_vertices - 1].position - scene_buffer_cpu[n_vertices - 2].position,
                scene_buffer_cpu[n_vertices - 3].position - scene_buffer_cpu[n_vertices - 2].position
            );

            normal = normal / (glm::length(normal) + 1e-6f); // Avoid division by zero

            scene_buffer_cpu[n_vertices - 1].normal = normal;
            scene_buffer_cpu[n_vertices - 2].normal = normal;
            scene_buffer_cpu[n_vertices - 3].normal = normal;

            index_offset += fv;
            // per-face material
            shapes[s].mesh.material_ids[f];
        }
    }

    scene_buffer->Store(scene_buffer_cpu.data(), n_vertices * sizeof(Scene::SceneVertex));
}

GLRenderer::~GLRenderer() 
{
    glDeleteTextures(1, &g_position_texture);
    glDeleteTextures(1, &g_normal_texture);
    glDeleteTextures(1, &g_depth_texture);
    glDeleteFramebuffers(1, &g_framebuffer);
    glDeleteFramebuffers(1, &deferred_framebuffer);
}

void GLRenderer::Render(Camera &camera) 
{
    glm::mat4 view_matrix = camera.GetViewMatrix();
    glm::mat4 projection_matrix = camera.GetProjectionMatrix();

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);    
    glFrontFace(GL_CCW);

    glBindFramebuffer(GL_FRAMEBUFFER, g_framebuffer);
    glViewport(0, 0, width, height);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    prepass_vao.Bind();
    prepass_program->SetMat4("view", 1, view_matrix);
    prepass_program->SetMat4("projection", 1, projection_matrix);
    prepass_program->DrawArrays(GL_TRIANGLES, 0, n_vertices);


    glBindFramebuffer(GL_FRAMEBUFFER, deferred_framebuffer);
    glViewport(0, 0, width, height);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    deferred_vao.Bind();
    deferred_program->SetVec3("camera_position", 1, camera.GetPosition());
    deferred_program->DrawArrays(GL_TRIANGLES, 0, 3);
}