#pragma once

#include "utils/opengl/buffer.hpp"

#include <glad/glad.h>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ranges.h>


namespace OpenGL
{

struct VertexAttribute
{
    public:

        VertexAttribute(GLenum type, GLint size, GLboolean normalized)
        {
            std::vector<GLenum> types{GL_BYTE, GL_UNSIGNED_BYTE, GL_SHORT, GL_UNSIGNED_SHORT, GL_INT, GL_UNSIGNED_INT, GL_HALF_FLOAT, GL_FLOAT, GL_FIXED, GL_DOUBLE};
            if (std::find(types.begin(), types.end(), type) == types.end())
            {
                SPDLOG_ERROR("type {} not valid", type);
                SPDLOG_ERROR("possible types {}", types);
                return;
            }

            std::vector<GLint> sizes{1, 2, 3, 4};
            if (std::find(sizes.begin(), sizes.end(), size) == sizes.end())
            {
                SPDLOG_ERROR("size {} not valid", size);
                SPDLOG_ERROR("possible sizes", sizes);
                return;
            }

            this->type = type;
            this->size = size;
            this->normalized = normalized;
        }

        GLenum Type()
        {
            return type;
        }

        GLint Size()
        {
            return size;
        }

        GLboolean Normalized()
        {
            return normalized;
        }

        GLuint Bytes()
        {
            switch(type)
            {
                case GL_BYTE:              return 1 * size;
                case GL_UNSIGNED_BYTE:     return 1 * size;
                case GL_SHORT:             return 2 * size;
                case GL_UNSIGNED_SHORT:    return 2 * size;
                case GL_INT:               return 4 * size;
                case GL_UNSIGNED_INT:      return 4 * size;

                case GL_HALF_FLOAT:        return 2 * size;
                case GL_FLOAT:             return 4 * size;
                case GL_FIXED:             return 4 * size;
    
                case GL_DOUBLE:            return 8 * size;
            };
            return 0;
        }

    private:

        GLenum type;
        GLint size;
        GLboolean normalized;
};


class VertexArray
{
    public:
        VertexArray()
        {
            glCreateVertexArrays(1, &handle);
        }

        ~VertexArray()
        {
            glDeleteVertexArrays(1, &handle);
        }

        VertexArray(const VertexArray&) = delete;
        VertexArray& operator=(const VertexArray&) = delete;


        GLuint Handle()
        {
            return handle;
        }

        GLuint Stride()
        {
            return stride;
        }

        void Bind()
        {
            glBindVertexArray(handle);
        }

        void Unbind()
        {
            glBindVertexArray(0);
        }

        void SetIndexBuffer(Buffer &buffer)
        {
            glVertexArrayElementBuffer(handle, buffer.Handle());
        }

        void SetVertexBufferLayout(std::vector<VertexAttribute> layout)
        {           
            if(has_layout)
            {
                stride = 0;
                for(int i = 0; i < this->layout.size(); i++)
                {
                    glDisableVertexArrayAttrib(handle, i);
                }
            }

            has_layout = true;
            this->layout = layout;

            stride = 0;
            for(int i = 0; i < layout.size(); i++)
            {
                glEnableVertexArrayAttrib(handle, i);
                glVertexArrayAttribBinding(handle, i, 0);

                auto &attribute = layout[i];
                if (std::find(float_types.begin(), float_types.end(), attribute.Type()) == float_types.end())
                {
                    glVertexArrayAttribIFormat(handle, i, attribute.Size(), attribute.Type(), stride);
                }
                else if (std::find(int_types.begin(), int_types.end(), attribute.Type()) == int_types.end())
                {
                    glVertexArrayAttribFormat(handle, i, attribute.Size(), attribute.Type(), attribute.Normalized(), stride);
                }
                else if (std::find(long_types.begin(), long_types.end(), attribute.Type()) == long_types.end())
                {
                    glVertexArrayAttribLFormat(handle, i, attribute.Size(), attribute.Type(), stride);
                }

                stride += attribute.Bytes();
            }
            Commit();
        }

        void SetVertexBuffer(BufferBase &buffer, GLintptr offset = 0)
        {           
            has_buffer = true;
            buffer_handle = buffer.Handle();
            buffer_offset = offset;
            Commit();
        }

        void SetVertexBufferAndLayout(BufferBase &buffer, std::vector<VertexAttribute> layout)
        {           
            SetVertexBufferLayout(layout);
            SetVertexBuffer(buffer);
        }

        void UnsetVertexBufferLayout()
        {
            has_layout = false;
            stride = 0;

            for(int i = 0; i < layout.size(); i++)
            {
                glDisableVertexArrayAttrib(handle, i);
            }
            
            layout.clear();
            Commit();
        }

        void UnsetVertexBuffer()
        {   
            has_buffer = false;
            buffer_handle = 0;
            buffer_offset = 0;
            Commit();
        }

        void UnsetVertexBufferAndLayout()
        {           
            UnsetVertexBufferLayout();
            UnsetVertexBuffer();
        }

    private:

        void Commit()
        {
            if(has_layout)
            {
                glVertexArrayVertexBuffer(handle, 0, buffer_handle, buffer_offset, stride);
            }
        }

        static inline std::vector<GLenum> int_types{GL_BYTE, GL_UNSIGNED_BYTE, GL_SHORT, GL_UNSIGNED_SHORT, GL_INT, GL_UNSIGNED_INT};
        static inline std::vector<GLenum> float_types{GL_HALF_FLOAT, GL_FLOAT, GL_FIXED};
        static inline std::vector<GLenum> long_types{GL_DOUBLE};


        GLuint handle;
        bool has_layout = false;
        bool has_buffer = false;

        GLuint buffer_handle = 0;
        GLintptr buffer_offset = 0;
        std::vector<VertexAttribute> layout;
        GLuint stride = 0;

        inline static int counter = 0;
};

}
