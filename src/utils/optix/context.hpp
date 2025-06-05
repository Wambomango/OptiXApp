#pragma once

#include <optix.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>

#include "./utils.hpp"

namespace OptiX
{
    class Context
    {
    public:
        Context()
        {
            CUDA_CHECK(cudaFree(0));
            OPTIX_CHECK(optixInitWithHandle(&handle));

            OptixDeviceContextOptions options = {};
            options.logCallbackFunction = &context_log_cb;
            options.logCallbackLevel = 4;
            OPTIX_CHECK(optixDeviceContextCreate(0, &options, &context));
        }

        ~Context()
        {
            OPTIX_CHECK(optixDeviceContextDestroy(context));
            OPTIX_CHECK(optixUninitWithHandle(handle));
        }

        OptixDeviceContext Handle()
        {
            return context;
        }

    private:
        static void context_log_cb(unsigned int level, const char *tag, const char *message, void * /*cbdata */)
        {
            switch (level)
            {
            case 0:
                return;

            case 1:
                SPDLOG_CRITICAL("[{}]: {}", tag, message);
                return;

            case 2:
                SPDLOG_ERROR("[{}]: {}", tag, message);
                return;

            case 3:
                SPDLOG_WARN("[{}]: {}", tag, message);
                return;

            case 4:
                SPDLOG_INFO("[{}]: {}", tag, message);
                return;

            default:
                SPDLOG_DEBUG("[{}]: {}", tag, message);
                break;
            }
        }

        void *handle;
        OptixDeviceContext context = nullptr;
    };

}