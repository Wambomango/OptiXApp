#pragma once

#include <cuda_runtime_api.h>
#include <nvrtc.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <exception>
#include <fstream>

#include <spdlog/spdlog.h>

#define MA_INCLUDE_DIRS                                      \
    "/home/mario/Desktop/Masterarbeit/OptiXApp/src/modules", \
    "/home/mario/Desktop/Masterarbeit/OptiXApp/dependencies/OptiX",

#define MA_OPTIONS         \
    "-std=c++11",          \
        "-arch",           \
        "compute_75",      \
        "-use_fast_math",  \
        "-default-device", \
        "-rdc",            \
        "true",            \
        "-D__x86_64",      \
        "-optix-ir",

#define OPTIX_CHECK(call) Utils::optixCheck(call, #call, __FILE__, __LINE__)
#define OPTIX_CHECK_LOG(call)                                                              \
    {                                                                                      \
        char LOG[2048];                                                                    \
        size_t LOG_SIZE = sizeof(LOG);                                                     \
        Utils::optixCheckLog(call, LOG, sizeof(LOG), LOG_SIZE, #call, __FILE__, __LINE__); \
    }

#define CUDA_CHECK(call) Utils::cudaCheck(call, #call, __FILE__, __LINE__)

#define NVRTC_CHECK_ERROR(func)                                                                                                                          \
    do                                                                                                                                                   \
    {                                                                                                                                                    \
        nvrtcResult code = func;                                                                                                                         \
        if (code != NVRTC_SUCCESS)                                                                                                                       \
            throw std::runtime_error(std::string("ERROR: ") + __FILE__ "(" + std::to_string(__LINE__) + "): " + std::string(nvrtcGetErrorString(code))); \
    } while (0);

namespace Utils
{
    inline void optixCheck(OptixResult res, const char *call, const char *file, unsigned int line)
    {
        if (res != OPTIX_SUCCESS)
        {
            std::stringstream ss;
            ss << "Optix call (" << call << ") failed: " << file << ':' << line << ")\n";
            throw std::runtime_error(ss.str().c_str());
        }
    };

    inline void optixCheckLog(OptixResult res, const char *log, size_t sizeof_log, size_t sizeof_log_returned, const char *call, const char *file, unsigned int line)
    {
        if (res != OPTIX_SUCCESS)
        {
            std::stringstream ss;
            ss << "Optix call '" << call << "' failed: " << file << ':' << line << ")\nLog:\n"
               << log << (sizeof_log_returned > sizeof_log ? "<TRUNCATED>" : "") << '\n';
            throw std::runtime_error(ss.str().c_str());
        }
    }

    inline void cudaCheck(cudaError_t error, const char *call, const char *file, unsigned int line)
    {
        if (error != cudaSuccess)
        {
            std::stringstream ss;
            ss << "CUDA call (" << call << " ) failed with error: '"
               << cudaGetErrorString(error) << "' (" << file << ":" << line << ")\n";
            throw std::runtime_error(ss.str().c_str());
        }
    }

    inline std::string ReadFile(const std::string &filename)
    {
        std::ifstream file(filename);
        if (!file.is_open())
        {
            throw std::runtime_error("Could not open file: " + filename);
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }

    inline std::string CudaToOptiXIR(const std::string &name, const std::string &content, std::vector<const char *> include_dirs = {MA_INCLUDE_DIRS}, std::vector<const char *> compile_options = {MA_OPTIONS})
    {
        std::vector<std::string> options;
        for (const char *dir : include_dirs)
        {
            options.push_back(std::string("-I") + dir);
        }

        for (const char *compile_option : compile_options)
        {
            options.push_back(compile_option);
        }

        std::vector<const char *> options_cstr;
        for (auto &option : options)
        {
            options_cstr.push_back(option.c_str());
        }

        nvrtcProgram prog = 0;
        NVRTC_CHECK_ERROR(nvrtcCreateProgram(&prog, content.c_str(), name.c_str(), 0, NULL, NULL));
        const nvrtcResult compileRes = nvrtcCompileProgram(prog, (int)options_cstr.size(), options_cstr.data());

        // Retrieve log output
        size_t log_size = 0;
        NVRTC_CHECK_ERROR(nvrtcGetProgramLogSize(prog, &log_size));

        std::string g_nvrtcLog;
        g_nvrtcLog.resize(log_size);
        if (log_size > 1)
        {
            NVRTC_CHECK_ERROR(nvrtcGetProgramLog(prog, &g_nvrtcLog[0]));
        }
        if (compileRes == NVRTC_SUCCESS)
        {
            SPDLOG_INFO("NVRTC Compilation successful for '{}'", name);
        }
        else
        {
            SPDLOG_ERROR("NVRTC Compilation failed for '{}'", name);
            NVRTC_CHECK_ERROR(compileRes);
        }

        // Retrieve OPTIXIR/PTX code
        size_t n_bytes;
        std::string bytecode;
        NVRTC_CHECK_ERROR(nvrtcGetOptiXIRSize(prog, &n_bytes));
        bytecode.resize(n_bytes);
        NVRTC_CHECK_ERROR(nvrtcGetOptiXIR(prog, &bytecode[0]));
        NVRTC_CHECK_ERROR(nvrtcDestroyProgram(&prog));

        return bytecode;
    }
}
