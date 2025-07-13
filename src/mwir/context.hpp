#pragma once

#include "utils/optix/context.hpp"

namespace MWIR
{
    class Context
    {
    public:
        static OptiX::Context& GetInstance()
        {
            static OptiX::Context instance;
            return instance;
        }

        Context(const Context&) = delete;
        Context& operator=(const Context&) = delete;

    private:
        Context() = default;
        ~Context() = default;
        Context(Context&&) = delete;
        Context& operator=(Context&&) = delete; 
    };
}