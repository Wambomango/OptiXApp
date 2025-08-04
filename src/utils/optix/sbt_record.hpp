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

        SBTRecord(Context &context, std::vector<OptixProgramGroup> &pg_handles, std::vector<T> records_data)
        {
            if(pg_handles.empty())
            {
                throw std::runtime_error("SBTRecord: No program groups provided.");
            }
            if(records_data.empty())
            {
                throw std::runtime_error("SBTRecord: No records provided.");
            }
            if(pg_handles.size() != records_data.size())
            {
                throw std::runtime_error("SBTRecord: Number of program groups and records data must match.");
            }


            n_elements = static_cast<int>(pg_handles.size());

            SBTRecordStruct record_structs[n_elements];
            for(size_t i = 0; i < n_elements; i++)
            {
                record_structs[i].data = records_data[i];
                memset(record_structs[i].header, 0, OPTIX_SBT_RECORD_HEADER_SIZE);
                OPTIX_CHECK(optixSbtRecordPackHeader(pg_handles[i], &record_structs[i]));
            }

            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_ptr), n_elements * sizeof(SBTRecordStruct)));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(device_ptr), (void *)record_structs, n_elements * sizeof(SBTRecordStruct), cudaMemcpyHostToDevice));
        }

        ~SBTRecord()
        {
            CUDA_CHECK(cudaFree(reinterpret_cast<void *>(device_ptr)));
        }

        CUdeviceptr Handle()
        {
            return device_ptr;
        }

        int NumElements()
        {
            return n_elements;
        }

        size_t ElementSize()
        {
            return sizeof(SBTRecordStruct);
        }

    private:
        CUdeviceptr device_ptr;
        int n_elements;
    };
}