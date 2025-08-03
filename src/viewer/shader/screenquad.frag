#include "bindings"


layout(location = 0) in vec2 uv;

layout (location = 0) out vec4 fragColor;

layout(binding = TEXTURE_UNIT_OUTPUT) uniform sampler2D cuda_texture;

void main() 
{
    fragColor = texture(cuda_texture, uv);
}

