#pragma once

#include "context.hpp"

namespace OptiX
{
    template <typename T>
    class SBTRecord
    {
    public:
        SBTRecord(Context &context, ProgramGroup &prog_group, T record = {})
        {
            OPTIX_CHECK(optixSbtRecordPackHeader(prog_group.Handle(), &record));

            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_ptr), sizeof(T)));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(device_ptr), &record, sizeof(T), cudaMemcpyHostToDevice));    
        }

        ~SBTRecord()
        {
            CUDA_CHECK(cudaFree(reinterpret_cast<void *>(device_ptr)));
        }

        CUdeviceptr Handle()
        {
            return device_ptr;
        }

    private:
        CUdeviceptr device_ptr;
    };
}