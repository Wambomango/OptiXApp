#include "gl_renderer.hpp"


GLRenderer::GLRenderer(Window &window, Scene& scene) 
{
    width = window.GetWidth();
    height = window.GetHeight();
    output_texture = window.GetTexture();

    glCreateTextures(GL_TEXTURE_2D, 1, &scene_depth_texture);
    glActiveTexture(GL_TEXTURE0 + 2);
    glTextureStorage2D(scene_depth_texture, 1, GL_DEPTH_COMPONENT32F, width, height);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glCreateFramebuffers(1, &scene_framebuffer);
    glNamedFramebufferTexture(scene_framebuffer, GL_COLOR_ATTACHMENT0, window.GetTexture(), 0);
    glNamedFramebufferTexture(scene_framebuffer, GL_DEPTH_ATTACHMENT, scene_depth_texture, 0);
    glCheckNamedFramebufferStatus(scene_framebuffer, GL_FRAMEBUFFER);

    scene_buffer = std::make_unique<OpenGL::Buffer>(1000 * 1000 * 1000, GL_STATIC_DRAW);
    scene_vao.SetVertexBufferAndLayout(*scene_buffer, {{GL_FLOAT, 3, false}, {GL_FLOAT, 3, false}});
    scene_program = std::make_unique<OpenGL::Program>(SHADER_DIR + std::string("scene.vert"), SHADER_DIR + std::string("scene.frag"));

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

            glm::vec3 normal = glm::normalize(glm::cross(
                scene_buffer_cpu[n_vertices - 1].position - scene_buffer_cpu[n_vertices - 2].position,
                scene_buffer_cpu[n_vertices - 3].position - scene_buffer_cpu[n_vertices - 2].position
            ));

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
    glDeleteFramebuffers(1, &scene_framebuffer);   
}    

void GLRenderer::Render(Camera &camera) 
{
    glEnable(GL_DEPTH_TEST);
    // glDepthFunc(GL_GEQUAL);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);    
    glFrontFace(GL_CCW);

    glBindFramebuffer(GL_FRAMEBUFFER, scene_framebuffer);
    glViewport(0, 0, width, height);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    // glClearDepth(0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


    scene_vao.Bind();

    glm::mat4 view_matrix = camera.GetViewMatrix();
    glm::mat4 projection_matrix = camera.GetProjectionMatrix();
    scene_program->SetMat4("view", 1, view_matrix);
    scene_program->SetMat4("projection", 1, projection_matrix);
    scene_program->SetVec3("camera_position", 1, camera.GetPosition());
    scene_program->DrawArrays(GL_TRIANGLES, 0, n_vertices);
}