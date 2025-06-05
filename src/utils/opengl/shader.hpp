#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <filesystem>
#include <sstream>
#include <fstream>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ranges.h>


#define SHADER_DIR "/home/mario/Desktop/Masterarbeit/OptiXApp/src/shader/"

#define SHADERTYPESTRING(t) #t

namespace OpenGL
{

class Shader
{
    public:
        Shader(const std::string filename)
        {
            ParseFilename(filename);
            if (failed)
            {
                return;
            }
            code = RecursiveReadFile(filename);
            if (failed)
            {
                return;
            }

            SPDLOG_TRACE(code);

            handle = glCreateShader(type);

            const char *wrapped_code[] = {"#version 460\n", code.c_str(), NULL};
            glShaderSource(handle, 2, &wrapped_code[0], 0);
            glCompileShader(handle);

            GLint status;
            GLsizei length = 0;
            char buffer[8192];
            glGetShaderiv(handle, GL_COMPILE_STATUS, &status);
            glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &length);
            glGetShaderInfoLog(handle, 8192, 0, buffer);

            if (length > 0)
            {
                SPDLOG_ERROR("{}", buffer);
            }
            if (status == GL_TRUE)
            {
                SPDLOG_DEBUG("compiled successfully");
            }
            else
            {
                SPDLOG_ERROR("failed to compile");
            }
        }

        ~Shader()
        {
            glDeleteShader(handle);
        }

        GLenum type = GL_INVALID_ENUM;
        GLuint handle;

    private:
        bool failed = false;
        std::string code;

        void ParseFilename(std::string filename)
        {
            if (!std::filesystem::exists(filename))
            {
                failed = true;
                SPDLOG_ERROR("file does not exist");
                return;
            }

            std::filesystem::path p(filename);
            std::string extension = p.extension();

            if (extension == ".vert")
            {
                type = GL_VERTEX_SHADER;
            }
            else if (extension == ".frag")
            {
                type = GL_FRAGMENT_SHADER;
            }
            else if (extension == ".geom")
            {
                type = GL_GEOMETRY_SHADER;
            }
            else if (extension == ".tesc")
            {
                type = GL_TESS_CONTROL_SHADER;
            }
            else if (extension == ".tese")
            {
                type = GL_TESS_EVALUATION_SHADER;
            }
            else if (extension == ".comp")
            {
                type = GL_COMPUTE_SHADER;
            }
            else
            {
                failed = true;
                SPDLOG_ERROR("invalid extension \"{}\"", extension);
            }
        }

        std::string RecursiveReadFile(std::string filename)
        {
            std::filesystem::path p(filename);
            std::string directory = p.parent_path().string() + "/";

            std::ifstream file;
            file.open(filename);
            if (file.fail())
            {
                failed = true;
                SPDLOG_ERROR("could not read {}", filename);
                return "";
            }

            std::stringstream result;
            std::string line;
            while (std::getline(file, line)) 
            {
                std::size_t line_begin = 0;
                while (line_begin < line.size() && std::isspace(static_cast<unsigned char>(line[line_begin]))) 
                {
                    line_begin++;
                }

                if (line.compare(line_begin, 9, "#include ") == 0 && line.back() == '"') 
                {
                    // Extract the filename between quotes
                    std::size_t quote_start = line.find('"', line_begin + 8);
                    std::size_t quote_end = line.rfind('"');
                    if (quote_start != std::string::npos && quote_end != std::string::npos && quote_start < quote_end) 
                    {
                        std::string include_file = line.substr(quote_start + 1, quote_end - quote_start - 1);
                        
                        result << RecursiveReadFile(directory + include_file);
                        continue;
                    }
                }
                else
                {
                    result << line << '\n';
                }
            }
        
            return result.str();
        }

    private:
        inline static int counter = 0;
};

class Program
{
    public:
        Program()
        {


        }

        Program(Shader &s1) 
        {
            handle = glCreateProgram();
            glAttachShader(handle, s1.handle);
            glLinkProgram(handle);
            CheckStatus();
            SPDLOG_DEBUG("initialized");
        }

        Program(std::string s1) 
        {
            handle = glCreateProgram();
            shaders.emplace_back(s1);
            glAttachShader(handle, shaders.back().handle);
            glLinkProgram(handle);
            CheckStatus();
            SPDLOG_DEBUG("initialized");
        }

        Program(Shader &s1, Shader &s2) 
        {
            handle = glCreateProgram();
            glAttachShader(handle, s1.handle);
            glAttachShader(handle, s2.handle);
            glLinkProgram(handle);
            CheckStatus();
            SPDLOG_DEBUG("initialized");
        }

        Program(std::string s1, std::string s2)
        {
            handle = glCreateProgram();
            shaders.emplace_back(s1);
            glAttachShader(handle, shaders.back().handle);
            shaders.emplace_back(s2);
            glAttachShader(handle, shaders.back().handle);
            glLinkProgram(handle);
            CheckStatus();
            SPDLOG_DEBUG("initialized");
        }

        Program(Shader &s1, Shader &s2, Shader &s3) 
        {
            handle = glCreateProgram();
            glAttachShader(handle, s1.handle);
            glAttachShader(handle, s2.handle);
            glAttachShader(handle, s3.handle);
            glLinkProgram(handle);
            CheckStatus();
            SPDLOG_DEBUG("initialized");
        }

        Program(std::string s1, std::string s2, std::string s3)
        {
            handle = glCreateProgram();
            shaders.emplace_back(s1);
            glAttachShader(handle, shaders.back().handle);
            shaders.emplace_back(s2);
            glAttachShader(handle, shaders.back().handle);
            shaders.emplace_back(s3);
            glAttachShader(handle, shaders.back().handle);
            glLinkProgram(handle);
            CheckStatus();
            SPDLOG_DEBUG("initialized");
        }

        Program(Shader &s1, Shader &s2, Shader &s3, Shader &s4)
        {
            handle = glCreateProgram();
            glAttachShader(handle, s1.handle);
            glAttachShader(handle, s2.handle);
            glAttachShader(handle, s3.handle);
            glAttachShader(handle, s4.handle);
            glLinkProgram(handle);
            CheckStatus();
            SPDLOG_DEBUG("initialized");
        }

        Program(std::string s1, std::string s2, std::string s3, std::string s4)
        {
            handle = glCreateProgram();
            shaders.emplace_back(s1);
            glAttachShader(handle, shaders.back().handle);
            shaders.emplace_back(s2);
            glAttachShader(handle, shaders.back().handle);
            shaders.emplace_back(s3);
            glAttachShader(handle, shaders.back().handle);
            shaders.emplace_back(s4);
            glAttachShader(handle, shaders.back().handle);
            glLinkProgram(handle);
            CheckStatus();
            SPDLOG_DEBUG("initialized");
        }

        Program(Shader &s1, Shader &s2, Shader &s3, Shader &s4, Shader &s5)
        {
            handle = glCreateProgram();
            glAttachShader(handle, s1.handle);
            glAttachShader(handle, s2.handle);
            glAttachShader(handle, s3.handle);
            glAttachShader(handle, s4.handle);
            glAttachShader(handle, s5.handle);
            glLinkProgram(handle);
            CheckStatus();
            SPDLOG_DEBUG("initialized");
        }

        Program(std::string s1, std::string s2, std::string s3, std::string s4, std::string s5)
        {
            handle = glCreateProgram();
            shaders.emplace_back(s1);
            glAttachShader(handle, shaders.back().handle);
            shaders.emplace_back(s2);
            glAttachShader(handle, shaders.back().handle);
            shaders.emplace_back(s3);
            glAttachShader(handle, shaders.back().handle);
            shaders.emplace_back(s4);
            glAttachShader(handle, shaders.back().handle);
            shaders.emplace_back(s5);
            glAttachShader(handle, shaders.back().handle);
            glLinkProgram(handle);
            CheckStatus();
            SPDLOG_DEBUG("initialized");
        }

        ~Program()
        {
            glDeleteProgram(handle);
        }

        void Use()
        {
            glUseProgram(handle);
        }

        void SetSSBO(std::string name, GLuint binding)
        {
            glShaderStorageBlockBinding(handle, glGetProgramResourceIndex(handle, GL_SHADER_STORAGE_BLOCK, name.c_str()), binding);
        }

        void SetBool(std::string name, GLint value)
        {
            glProgramUniform1i(handle, glGetUniformLocation(handle, name.c_str()), value);
        }

        void SetInt(std::string name, GLint value)
        {
            glProgramUniform1i(handle, glGetUniformLocation(handle, name.c_str()), value);
        }

        void SetUnsignedInt(std::string name, GLuint value)
        {
            glProgramUniform1ui(handle, glGetUniformLocation(handle, name.c_str()), value);
        }

        void SetFloat(std::string name, GLfloat value)
        {
            glProgramUniform1f(handle, glGetUniformLocation(handle, name.c_str()), value);
        }

        void SetVec2(std::string name, GLsizei count, glm::vec2 &value)
        {
            glProgramUniform2fv(handle, glGetUniformLocation(handle, name.c_str()), count, &value[0]);
        }

        void SetVec2(std::string name, GLfloat x, GLfloat y)
        {
            glProgramUniform2f(handle, glGetUniformLocation(handle, name.c_str()), x, y);
        }

        void SetVec3(std::string name, GLsizei count, glm::vec3 &value)
        {
            glProgramUniform3fv(handle, glGetUniformLocation(handle, name.c_str()), count, &value[0]);
        }

        void SetVec3(std::string name, GLfloat x, GLfloat y, GLfloat z)
        {
            glProgramUniform3f(handle, glGetUniformLocation(handle, name.c_str()), x, y, z);
        }

        void SetVec4(std::string name, GLsizei count, glm::vec4 &value)
        {
            glProgramUniform4fv(handle, glGetUniformLocation(handle, name.c_str()), count, &value[0]);
        }

        void SetVec4(std::string name, GLfloat x, GLfloat y, GLfloat z, GLfloat w)
        {
            glProgramUniform4f(handle, glGetUniformLocation(handle, name.c_str()), x, y, z, w);
        }

        void SetIntegerVec2(std::string name, GLsizei count, glm::ivec2 &value)
        {
            glProgramUniform2iv(handle, glGetUniformLocation(handle, name.c_str()), count, &value[0]);
        }

        void SetIntegerVec2(std::string name, GLint x, GLint y)
        {
            glProgramUniform2i(handle, glGetUniformLocation(handle, name.c_str()), x, y);
        }

        void SetIntegerVec3(std::string name, GLsizei count, glm::ivec3 &value)
        {
            glProgramUniform3iv(handle, glGetUniformLocation(handle, name.c_str()), count, &value[0]);
        }

        void SetIntegerVec3(std::string name, GLint x, GLint y, GLint z)
        {
            glProgramUniform3i(handle, glGetUniformLocation(handle, name.c_str()), x, y, z);
        }

        void SetIntegerVec4(std::string name, GLsizei count, glm::ivec4 &value)
        {
            glProgramUniform4iv(handle, glGetUniformLocation(handle, name.c_str()), count, &value[0]);
        }

        void SetIntegerVec4(std::string name, GLint x, GLint y, GLint z, GLint w)
        {
            glProgramUniform4i(handle, glGetUniformLocation(handle, name.c_str()), x, y, z, w);
        }

        void SetMat2(std::string name, GLsizei count, glm::mat2 &mat)
        {
            glProgramUniformMatrix2fv(handle, glGetUniformLocation(handle, name.c_str()), count, GL_FALSE, &mat[0][0]);
        }

        void SetMat3(std::string name, GLsizei count, glm::mat3 &mat)
        {
            glProgramUniformMatrix3fv(handle, glGetUniformLocation(handle, name.c_str()), count, GL_FALSE, &mat[0][0]);
        }

        void SetMat4(std::string name, GLsizei count, glm::mat4 &mat)
        {
            glProgramUniformMatrix4fv(handle, glGetUniformLocation(handle, name.c_str()), count, GL_FALSE, &mat[0][0]);
        }

        void DrawArrays(GLenum mode, GLint first, GLsizei count)
        {
            glUseProgram(handle);
            glDrawArrays(mode, first, count);
        }

        void DrawArraysInstancedBaseInstance(GLenum mode, GLint first, GLsizei count, GLsizei instancecount, GLuint baseinstance)
        {
            glUseProgram(handle);
            glDrawArraysInstancedBaseInstance(mode, first, count, instancecount, baseinstance);
        }

        void DrawArraysInstanced(GLenum mode, GLint first, GLsizei count, GLsizei instancecount)
        {
            glUseProgram(handle);
            glDrawArraysInstanced(mode, first, count, instancecount);
        }

        void DrawArraysIndirect(GLenum mode, const void *indirect)
        {
            glUseProgram(handle);
            glDrawArraysIndirect(mode, indirect);
        }

        void MultiDrawArrays(GLenum mode, const GLint *first, const GLsizei *count, GLsizei drawcount)
        {
            glUseProgram(handle);
            glMultiDrawArrays(mode, first, count, drawcount);
        }

        void MultiDrawArraysIndirect(GLenum mode, const void *indirect, GLsizei drawcount, GLsizei stride)
        {
            glUseProgram(handle);
            glMultiDrawArraysIndirect(mode, indirect, drawcount, stride);
        }

        void MultiDrawArraysIndirectCount(GLenum mode, const void *indirect, GLintptr drawcount, GLsizei maxdrawcount, GLsizei stride)
        {
            glUseProgram(handle);
            glMultiDrawArraysIndirectCount(mode, indirect, drawcount, maxdrawcount, stride);
        }

        void DrawElements(GLenum mode, GLsizei count, GLenum type, const void *indices)
        {
            glUseProgram(handle);
            glDrawElements(mode, count, type, indices);
        }

        void DrawElementsInstancedBaseInstance(GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLuint baseinstance)
        {
            glUseProgram(handle);
            glDrawElementsInstancedBaseInstance(mode, count, type, indices, instancecount, baseinstance);
        }

        void DrawElementsInstanced(GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount)
        {
            glUseProgram(handle);
            glDrawElementsInstanced(mode, count, type, indices, instancecount);
        }

        void MultiDrawElements(GLenum mode, const GLsizei *count, GLenum type, const void *const *indices, GLsizei drawcount)
        {
            glUseProgram(handle);
            glMultiDrawElements(mode, count, type, indices, drawcount);
        }

        void DrawRangeElements(GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices)
        {
            glUseProgram(handle);
            glDrawRangeElements(mode, start, end, count, type, indices);
        }

        void DrawElementsBaseVertex(GLenum mode, GLsizei count, GLenum type, const void *indices, GLint basevertex)
        {
            glUseProgram(handle);
            glDrawElementsBaseVertex(mode, count, type, indices, basevertex);
        }

        void DrawRangeElementsBaseVertex(GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices, GLint basevertex)
        {
            glUseProgram(handle);
            glDrawRangeElementsBaseVertex(mode, start, end, count, type, indices, basevertex);
        }

        void DrawElementsInstancedBaseVertex(GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLint basevertex)
        {
            glUseProgram(handle);
            glDrawElementsInstancedBaseVertex(mode, count, type, indices, instancecount, basevertex);
        }

        void DrawElementsInstancedBaseVertex(GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLint basevertex, GLuint baseinstance)
        {
            glUseProgram(handle);
            glDrawElementsInstancedBaseVertexBaseInstance(mode, count, type, indices, instancecount, basevertex, baseinstance);
        }

        void DrawElementsIndirect(GLenum mode, GLenum type, const void *indirect)
        {
            glUseProgram(handle);
            glDrawElementsIndirect(mode, type, indirect);
        }

        void MultiDrawElementsIndirect(GLenum mode, GLenum type, const void *indirect, GLsizei drawcount, GLsizei stride)
        {
            glUseProgram(handle);
            glMultiDrawElementsIndirect(mode, type, indirect, drawcount, stride);
        }

        void MultiDrawElementsIndirectCount(GLenum mode, GLenum type, const void *indirect, GLintptr drawcount, GLsizei maxdrawcount, GLsizei stride)
        {
            glUseProgram(handle);
            glMultiDrawElementsIndirectCount(mode, type, indirect, drawcount, maxdrawcount, stride);
        }

        void MultiDrawElementsBaseVertex(GLenum mode, const GLsizei *count, GLenum type, const void *const *indices, GLsizei drawcount, const GLint *basevertex)
        {
            glUseProgram(handle);
            glMultiDrawElementsBaseVertex(mode, count, type, indices, drawcount, basevertex);
        }

        void DispatchCompute(GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z)
        {
            glUseProgram(handle);
            glDispatchCompute(num_groups_x, num_groups_y, num_groups_z);
        }

    private:

        GLuint handle;
        std::vector<Shader> shaders;

        void CheckStatus()
        {
            GLint status;
            GLsizei length = 0;
            char buffer[8192];

            glGetProgramiv(handle, GL_LINK_STATUS, &status);
            glGetProgramiv(handle, GL_INFO_LOG_LENGTH, &length);
            glGetProgramInfoLog(handle, 8192, 0, buffer);

            if (length > 0)
            {
                SPDLOG_ERROR("{}", buffer);
            }
            if (status == GL_TRUE)
            {
                SPDLOG_DEBUG("linked successfully");
            }
            else
            {
                SPDLOG_ERROR("failed to link");
            }
        }

    private:
        inline static int counter = 0;
};

}
