#include "bindings"

layout(location = 0) in vec2 uv;

layout (location = 0) out float fragColor;

layout(binding = TEXTURE_UNIT_SSAO_RAW) uniform sampler2D ssao_raw_texture;

uniform int width;
uniform int height;

const int KERNEL_SIZE = 1;

void main() 
{

    vec2 texelSize = 1.0 / vec2(width, height);
    float result = 0.0;

    for (int x = -KERNEL_SIZE; x <= KERNEL_SIZE; ++x) 
    {
        for (int y = -KERNEL_SIZE; y <= KERNEL_SIZE; ++y) 
        {
            vec2 offset = vec2(float(x), float(y)) * texelSize;
            result += texture(ssao_raw_texture, uv + offset).x;
        }
    }

    fragColor = result / ((2 * KERNEL_SIZE + 1) * (2 * KERNEL_SIZE + 1));
}  



