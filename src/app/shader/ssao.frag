#include "bindings"

layout(location = 0) in vec2 uv;

layout (location = 0) out float fragColor;

layout(binding = TEXTURE_UNIT_POSITION) uniform sampler2D position_texture;
layout(binding = TEXTURE_UNIT_NORMAL) uniform sampler2D normal_texture;
layout(binding = TEXTURE_UNIT_DEPTH) uniform sampler2D depth_texture;


uniform int n_samples;
uniform int width;
uniform int height;
uniform mat4 view;
uniform mat4 projection;
uniform float near_plane;
uniform float far_plane;

layout(binding = SSBO_SAMPLES, std430) buffer ssbo_samples
{
    float samples[];
};

const int N_SAMPLES = 256;

float linearize_depth(float d, float zNear, float zFar)
{
    return zNear * zFar / (zFar + d * (zNear - zFar));
}

void main() 
{
    vec4 position = texture(position_texture, uv);
    vec3 normal = texture(normal_texture, uv).xyz;

    if(position.w == 0)
    {
        fragColor = 1;
        discard;
    }

    int offset = int((uv.x * width * height + uv.y * height)) % (3 * n_samples);

    vec3 randomVec = normalize(vec3(samples[offset + 0], samples[offset + 1], samples[offset + 2]));
    vec3 tangent   = normalize(randomVec - normal * dot(randomVec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN = mat3(tangent, bitangent, normal);  

    float radius = 0.5;
    float occlusion = 0.0;
    for(int i = 0; i < N_SAMPLES; i++)
    {
        vec3 s = vec3(samples[offset + i * 3 + 0], samples[offset + i * 3 + 1], samples[offset + i * 3 + 2]);
        s.z = abs(s.z);

        vec3 world_coordinates = position.xyz + TBN * s * radius;
        vec4 screen_coordinates = projection * view * vec4(world_coordinates, 1.0);
        vec3 uvdepth = screen_coordinates.xyz / screen_coordinates.w;
        float u = uvdepth.x * 0.5 + 0.5;
        float v = uvdepth.y * 0.5 + 0.5;
        if(u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0)
        {
            continue;
        }

        float image_depth = texture(depth_texture, vec2(u, v)).x;
        float sample_depth = uvdepth.z * 0.5 + 0.5;

        float image_depth_l = linearize_depth(image_depth, near_plane, far_plane);
        float sample_depth_l = linearize_depth(sample_depth, near_plane, far_plane);

        if(image_depth_l <= sample_depth_l + 0.0025)
        {   
            occlusion += smoothstep(0.0, 1.0, radius / abs(image_depth_l - sample_depth_l));
        }
    }

    fragColor = 1 - occlusion / N_SAMPLES;
}

