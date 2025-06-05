layout(location = 0) in vec3 normal;
layout(location = 1) in vec3 frag_pos;

layout (location = 0) out vec4 fragColor;

vec3 lightVector = normalize(vec3(-1.0, -1.0, -1.0));
vec3 lightColor = vec3(1.0, 1.0, 1.0); 

uniform vec3 camera_position;

void main() 
{
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

