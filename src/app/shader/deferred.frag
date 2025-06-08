#include "bindings"


layout(location = 0) in vec2 uv;

layout (location = 0) out vec4 fragColor;

layout(binding = TEXTURE_UNIT_POSITION) uniform sampler2D position_texture;
layout(binding = TEXTURE_UNIT_NORMAL) uniform sampler2D normal_texture;
layout(binding = TEXTURE_UNIT_SSAO) uniform sampler2D ssao_texture;

uniform vec3 light_direction;
uniform vec3 light_color; 
uniform vec3 camera_position;

void main() 
{
    vec4 frag_pos = texture(position_texture, uv);
    vec3 normal = texture(normal_texture, uv).xyz;
    float ssao = texture(ssao_texture, uv).x;

    if(frag_pos.w == 0.0) 
    {
        discard;
    }



    float ambientStrength = 0.5;
    vec3 ambient = ambientStrength * light_color;

    float diff = max(dot(normal, light_direction), 0.0);
    vec3 diffuse = diff * light_color;

    vec3 light = ambient + diffuse;


    float specularStrength = 0.5;
    if(dot(light_direction, normal) < 0.0) 
    {
        vec3 viewDir = normalize(camera_position - frag_pos.xyz);
        vec3 reflectDir = reflect(light_direction, normal);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 1);
        vec3 specular = specularStrength * spec * light_color;
        light += specular;
    }

    fragColor = vec4(light * ssao, 1.0);
}

