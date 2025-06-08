layout(location = 0) in vec3 frag_pos;
layout(location = 1) in vec3 normal;

layout (location = 0) out vec4 frag_pos_;
layout (location = 1) out vec3 normal_;


void main() 
{
    frag_pos_ = vec4(frag_pos, 1.0);
    normal_ = normal;
}


