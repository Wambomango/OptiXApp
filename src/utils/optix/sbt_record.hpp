#pragma once

#include "context.hpp"



namespace OptiX
{
    template <typename T>
    class SBTRecord
    {
    public:
        struct SBTRecordStruct
        {
            __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
            T data;
        };


        SBTRecord(Context &context, ProgramGroup &prog_group, T record = {})
        {
            SBTRecordStruct record_struct;
            record_struct.data = record;
            memset(record_struct.header, 0, OPTIX_SBT_RECORD_HEADER_SIZE);
            OPTIX_CHECK(optixSbtRecordPackHeader(prog_group.Handle(), &record_struct));

            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_ptr), sizeof(SBTRecordStruct)));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(device_ptr), &record_struct, sizeof(SBTRecordStruct), cudaMemcpyHostToDevice));
        }

        ~SBTRecord()
        {
            CUDA_CHECK(cudaFree(reinterpret_cast<void *>(device_ptr)));
        }

        CUdeviceptr Handle()
        {
            return device_ptr;
        }

        size_t Size()
        {
            return sizeof(SBTRecordStruct);
        }


    private:
        CUdeviceptr device_ptr;
    };
}