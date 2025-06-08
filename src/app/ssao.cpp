#include "ssao.hpp"

#define N_SAMPLES 1000000

SSAO::SSAO(int width, int height) : width(width), height(height)
{
    GL_CREATE_TEXTURE_2D(ssao_raw_texture, GL_R8, width, height);
    glBindTextureUnit(TEXTURE_UNIT_SSAO_RAW, ssao_raw_texture);

    GL_CREATE_TEXTURE_2D(ssao_texture, GL_R8, width, height);
    glBindTextureUnit(TEXTURE_UNIT_SSAO, ssao_texture);

    glCreateFramebuffers(1, &ssao_raw_framebuffer);
    glNamedFramebufferTexture(ssao_raw_framebuffer, GL_COLOR_ATTACHMENT0, ssao_raw_texture, 0);
    glCheckNamedFramebufferStatus(ssao_raw_framebuffer, GL_FRAMEBUFFER);

    glCreateFramebuffers(1, &ssao_framebuffer);
    glNamedFramebufferTexture(ssao_framebuffer, GL_COLOR_ATTACHMENT0, ssao_texture, 0);
    glCheckNamedFramebufferStatus(ssao_framebuffer, GL_FRAMEBUFFER);
    
    ssao_program = std::make_unique<OpenGL::Program>(SHADER_DIR + std::string("ssao.vert"), SHADER_DIR + std::string("ssao.frag"));
    ssao_program->SetInt("n_samples", N_SAMPLES);
    ssao_program->SetInt("width", width);
    ssao_program->SetInt("height", height);

    ssao_blur_program = std::make_unique<OpenGL::Program>(SHADER_DIR + std::string("ssao_blur.vert"), SHADER_DIR + std::string("ssao_blur.frag"));
    ssao_blur_program->SetInt("width", width);
    ssao_blur_program->SetInt("height", height);

    samples = std::make_unique<OpenGL::Buffer>(N_SAMPLES * sizeof(glm::vec4), GL_STATIC_DRAW);
    std::vector<glm::vec4> samples_cpu(N_SAMPLES);

    for(int i = 0; i < N_SAMPLES; i++) 
    {
        glm::vec4 sample;
        while(true)
        {
            sample.x = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
            sample.y = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
            sample.z = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
            if(glm::length(sample) < 1.0f) 
            {
                break;
            }
        }

        samples_cpu[i] = sample;
    }

    samples->Store(samples_cpu.data(), N_SAMPLES * sizeof(glm::vec4));    
    samples->BindRange(SSBO_SAMPLES);
}


SSAO::~SSAO()
{
    glDeleteTextures(1, &ssao_raw_texture);
    glDeleteTextures(1, &ssao_texture);
    glDeleteFramebuffers(1, &ssao_raw_framebuffer);
    glDeleteFramebuffers(1, &ssao_framebuffer);
}


void SSAO::CalculateSSAO(Camera &camera)
{
    glBindFramebuffer(GL_FRAMEBUFFER, ssao_raw_framebuffer);
    ssao_program->SetMat4("view", 1, camera.GetViewMatrix());
    ssao_program->SetMat4("projection", 1, camera.GetProjectionMatrix());
    ssao_program->SetFloat("near_plane", camera.GetNearPlane());
    ssao_program->SetFloat("far_plane", camera.GetFarPlane());
    ssao_program->DrawArrays(GL_TRIANGLES, 0, 3);

    glBindFramebuffer(GL_FRAMEBUFFER, ssao_framebuffer);
    ssao_blur_program->DrawArrays(GL_TRIANGLES, 0, 3);
}

