#pragma once

#include <glad/glad.h>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ranges.h>


namespace OpenGL
{

class BufferBase
{
    public:

        BufferBase()
        {
            glCreateBuffers(1, &handle);
        }

        BufferBase (const BufferBase&) = delete;
        BufferBase& operator= (const BufferBase&) = delete;

        virtual ~BufferBase()
        {
            if(mapped)
            {
                glUnmapNamedBuffer(handle);
            }
            glDeleteBuffers(1, &handle);
        }

        GLuint Handle()
        {
            return handle;
        }

        GLsizeiptr Capacity()
        {
            return capacity;
        }

        GLenum Usage()
        {
            return usage;
        }
        
        GLenum Mapped()
        {
            return mapped;
        }

        bool Store(void *source, GLsizeiptr size = 0xFFFFFFFFFF, GLintptr offset = 0)
        {
            size = std::min(size, capacity - offset);
            if (size > 0)
            {
                SPDLOG_TRACE("storing bytes [{},{}] to buffer", offset, offset + size - 1);
                glNamedBufferSubData(handle, offset, size, source);
                return true;
            }
            else
            {
                SPDLOG_ERROR("cannot store {}[bytes] to buffer", size);
                return false;
            }
        }

        bool Load(void *destination, GLsizeiptr size = 0xFFFFFFFFFF, GLintptr offset = 0)
        {
            size = std::min(size, capacity - offset);
            if (size > 0)
            {
                SPDLOG_TRACE("loading bytes [{},{}] from buffer", offset, offset + size - 1);
                glGetNamedBufferSubData(handle, offset, size, destination);
                return true;
            }
            else
            {
                SPDLOG_ERROR("cannot load {}[bytes] from buffer", size);
                return false;
            }
        }

        bool BindRange(GLuint index, GLsizeiptr size = 0xFFFFFFFFFF, GLintptr offset = 0, GLenum target = GL_SHADER_STORAGE_BUFFER)
        {
            size = std::min(size, capacity - offset);
            if (size > 0)
            {
                SPDLOG_TRACE("binding bytes [{},{}] to index {}", offset, offset + size - 1, index);
                glBindBufferRange(target, index, handle, offset, size);
                return true;
            }
            else
            {
                SPDLOG_ERROR("cannot bind {}[bytes] to index {}", size, index);
                return false;
            }
        }

        void BindTarget(GLenum target)
        {
            std::string target_string;

            switch(target)
            {
                case GL_ARRAY_BUFFER:                   target_string = "GL_ARRAY_BUFFER";
                                                        break;

                case GL_ATOMIC_COUNTER_BUFFER:          target_string = "GL_ATOMIC_COUNTER_BUFFER";
                                                        break;

                case GL_COPY_READ_BUFFER:               target_string = "GL_COPY_READ_BUFFER";
                                                        break;

                case GL_COPY_WRITE_BUFFER:              target_string = "GL_COPY_WRITE_BUFFER";
                                                        break;

                case GL_DISPATCH_INDIRECT_BUFFER:       target_string = "GL_DISPATCH_INDIRECT_BUFFER";
                                                        break;

                case GL_DRAW_INDIRECT_BUFFER:           target_string = "GL_DRAW_INDIRECT_BUFFER";
                                                        break;

                case GL_ELEMENT_ARRAY_BUFFER:           target_string = "GL_ELEMENT_ARRAY_BUFFER";
                                                        break;
                                    
                case GL_PIXEL_PACK_BUFFER:              target_string = "GL_PIXEL_PACK_BUFFER";
                                                        break;

                case GL_PIXEL_UNPACK_BUFFER:            target_string = "GL_PIXEL_UNPACK_BUFFER";
                                                        break;

                case GL_QUERY_BUFFER:                   target_string = "GL_QUERY_BUFFER";
                                                        break;

                case GL_SHADER_STORAGE_BUFFER:          target_string = "GL_SHADER_STORAGE_BUFFER";
                                                        break;

                case GL_TEXTURE_BUFFER:                 target_string = "GL_TEXTURE_BUFFER";
                                                        break;

                case GL_TRANSFORM_FEEDBACK_BUFFER:      target_string = "GL_TRANSFORM_FEEDBACK_BUFFER";
                                                        break;

                case GL_UNIFORM_BUFFER:                 target_string = "GL_UNIFORM_BUFFER";
                                                        break;
            };

            SPDLOG_TRACE("binding to {}", target_string);
            glBindBuffer(target, handle);
        }

        void *Map( GLenum access, GLsizeiptr length = 0xFFFFFFFFFF, GLintptr offset = 0)
        {
            length = std::min(length, capacity - offset);
            if (length > 0)
            {
                mapped = true;
                SPDLOG_TRACE("mapping bytes [{},{}]", offset, offset + length - 1);
                return glMapNamedBufferRange(handle, offset, length, access);
            }
            else
            {
                SPDLOG_ERROR("cannot bind {}[bytes]", length);
                return nullptr;
            }
        }

        bool Unmap()
        {
            if(mapped)
            {
                SPDLOG_TRACE("unmappping buffer");
                glUnmapNamedBuffer(handle);
                mapped = false;
                return true;
            }
            else
            {
                SPDLOG_ERROR("buffer is already unmapped");
                return false;
            }
        }

        bool Flush(GLsizeiptr length = 0xFFFFFFFFFF, GLintptr offset = 0)
        {
            length = std::min(length, capacity - offset);
            if (mapped && length > 0)
            {
                SPDLOG_TRACE("flushing bytes [{},{}]", offset, offset + length - 1);
                glFlushMappedNamedBufferRange(handle, offset, length);
                return true;
            }
            else
            {
                if(mapped)
                {
                    SPDLOG_ERROR("cannot flush {}[bytes]", length);
                }
                else
                {
                    SPDLOG_ERROR("unmapped buffer cannot be flushed");
                }

                return false;
            }
        }

        friend class Buffer;
        friend class IBuffer;

    protected:
        GLuint handle;
        GLsizeiptr capacity;
        GLenum usage;
        bool mapped = false;
};



class Buffer : public BufferBase
{
    public:
        Buffer() : BufferBase()
        {
            SPDLOG_DEBUG("initialized");
        }
        
        Buffer(GLsizeiptr capacity, GLenum usage, void *data = nullptr) : BufferBase()
        {
            Reallocate(capacity, usage, data);
            SPDLOG_DEBUG("initialized");
        }

        Buffer (const Buffer&) = delete;
        Buffer& operator= (const Buffer&) = delete;
        
        void Reallocate(GLsizeiptr capacity, GLenum usage, void *data = nullptr)
        {        
            this->capacity = capacity;
            this->usage = usage;

            SPDLOG_DEBUG("allocating {}[bytes]", capacity);
            glNamedBufferData(handle, capacity, nullptr, usage);
        }

    private:
        inline static int counter = 0;
};

class IBuffer : public BufferBase
{
    public:
        IBuffer(GLsizeiptr capacity, GLenum usage, void *data = nullptr) : BufferBase()
        {
            this->capacity = capacity;
            this->usage = usage;

            SPDLOG_DEBUG("allocating {}[bytes]", capacity);
            glNamedBufferStorage(handle, capacity, data, usage);

            SPDLOG_DEBUG("initialized");
        }

        IBuffer (const IBuffer&) = delete;
        IBuffer& operator= (const IBuffer&) = delete;
        

    private:
        inline static int counter = 0;
};

}

