layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

layout(location = 0) out vec3 normal_;
layout(location = 1) out vec3 frag_pos_;

uniform mat4 view;
uniform mat4 projection;

void main()
{
    normal_ = normal;
    frag_pos_ = position;

    gl_Position = projection * view * vec4(position, 1.0);
}