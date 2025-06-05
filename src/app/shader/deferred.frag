#include "bindings"


layout(location = 0) in vec2 uv;

layout (location = 0) out vec4 fragColor;

layout(binding = TEXTURE_UNIT_POSITION) uniform sampler2D position_texture;
layout(binding = TEXTURE_UNIT_NORMAL) uniform sampler2D normal_texture;



vec3 lightVector = normalize(vec3(-1.0, -1.0, -1.0));
vec3 lightColor = vec3(1.0, 1.0, 1.0); 
uniform vec3 camera_position;

void main() 
{
    vec3 frag_pos = texture(position_texture, uv).xyz;
    vec3 normal = texture(normal_texture, uv).xyz;

    float ambientStrength = 0.5;
    vec3 ambient = ambientStrength * lightColor;

    float diff = max(dot(normal, lightVector), 0.0);
    vec3 diffuse = diff * lightColor;

    float specularStrength = 0.5;
    vec3 viewDir = normalize(camera_position - frag_pos);
    vec3 reflectDir = reflect(-lightVector, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;

    fragColor = vec4(ambient + diffuse + specular, 1.0);
}

