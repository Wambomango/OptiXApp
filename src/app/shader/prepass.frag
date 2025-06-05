layout(location = 0) in vec3 frag_pos;
layout(location = 1) in vec3 normal;

layout (location = 0) out vec3 frag_pos_;
layout (location = 1) out vec3 normal_;


void main() 
{
    frag_pos_ = frag_pos;
    normal_ = normal;
}


