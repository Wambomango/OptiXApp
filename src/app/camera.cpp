#include "camera.hpp"

Camera::Camera(float fov, float aspect_ratio, float near_plane, float far_plane) : fov(fov), aspect_ratio(aspect_ratio), near_plane(near_plane), far_plane(far_plane)
{
    UpdateProjectionMatrix();
}

void Camera::Tick(float dt)
{
    float speed = fast ? 20.0f : 5.0f; // Speed of the camera movement
    glm::vec3 forward = glm::normalize(glm::vec3(
        glm::sin(glm::radians(orientation.x)),
        0.0f,
        -glm::cos(glm::radians(orientation.x))
    ));
    glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));
    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);

    if (moving_forward)
    {
        position += speed * dt * forward;
    }

    if (moving_backward)
    {
        position -= speed * dt * forward;
    }

    if (moving_right)
    {
        position += speed * dt * right;
    }

    if (moving_left)
    {
        position -= speed * dt * right;
    }

    if (moving_up)
    {
        position += speed * dt * up;
    }

    if (moving_down)
    {
        position -= speed * dt * up;
    }

    UpdateViewMatrix();
}

void Camera::AddCallbacks(Window& window)
{
    window.AddScrollCallback(std::bind(&Camera::OnScroll, this, std::placeholders::_1, std::placeholders::_2));
    window.AddMouseMoveCallback(std::bind(&Camera::OnMouseMove, this, std::placeholders::_1, std::placeholders::_2));
    window.AddMouseButtonCallback(std::bind(&Camera::OnMouseButton, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    window.AddKeyCallback(std::bind(&Camera::OnKey, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
}

void Camera::SetPosition(const glm::vec3& position)
{
    this->position = position;
    UpdateViewMatrix();
}
void Camera::SetOrientation(const glm::vec3& orientation)
{
    this->orientation = orientation;
    UpdateViewMatrix();
}
glm::vec3 Camera::GetPosition()
{
    return this->position;
}
glm::vec3 Camera::GetOrientation()
{
    return this->orientation;
}
glm::mat4 Camera::GetViewMatrix()
{
    return view_matrix;
}
glm::mat4 Camera::GetProjectionMatrix()
{
    return projection_matrix;
}
float Camera::GetNearPlane()
{
    return near_plane;
}
float Camera::GetFarPlane()
{
    return far_plane;
}
void Camera::UpdateProjectionMatrix()
{
    projection_matrix = glm::perspective(glm::radians(fov), aspect_ratio, near_plane, far_plane);
}
void Camera::UpdateViewMatrix()
{
    view_matrix = glm::eulerAngleX(glm::radians(-orientation.y)) * glm::eulerAngleY(glm::radians(orientation.x)) * glm::translate(glm::mat4(1.0f), -position);
}

float Camera::GetFOV()
{
    return fov;
}

float Camera::GetAspectRatio()
{
    return aspect_ratio;
}

void Camera::OnScroll(double xoffset, double yoffset)
{
    fov -= static_cast<float>(yoffset);
    fov = glm::clamp(fov, 1.0f, 45.0f); 
    UpdateProjectionMatrix();
}

void Camera::OnMouseMove(double xpos, double ypos)
{
    static double last_x = xpos;
    static double last_y = ypos;
    double xoffset = xpos - last_x;
    double yoffset = last_y - ypos; 
    last_x = xpos;
    last_y = ypos;
    float sensitivity = 0.2; 
    orientation.x += static_cast<float>(xoffset * sensitivity);
    orientation.y += static_cast<float>(yoffset * sensitivity);
    orientation.y = glm::clamp(orientation.y, -89.0f, 89.0f); 
    UpdateViewMatrix();
}

void Camera::OnMouseButton(int button, int action, int mods)
{
}

void Camera::OnKey(int key, int scancode, int action, int mods)
{            
    if (action == GLFW_PRESS)
    {
        switch (key)
        {
            case GLFW_KEY_W:
                moving_forward = true;
                break;
            case GLFW_KEY_S:
                moving_backward = true;
                break;
            case GLFW_KEY_D:
                moving_right = true;
                break;
            case GLFW_KEY_A:
                moving_left = true;
                break;
            case GLFW_KEY_SPACE:
                moving_up = true;
                break;
            case GLFW_KEY_LEFT_SHIFT:
                moving_down = true;
                break;
            case GLFW_KEY_LEFT_CONTROL:
                fast = true;
                break;
        }
    }
    else if (action == GLFW_RELEASE)
    {
        switch (key)
        {
            case GLFW_KEY_W:
                moving_forward = false;
                break;
            case GLFW_KEY_S:
                moving_backward = false;
                break;
            case GLFW_KEY_D:
                moving_right = false;
                break;
            case GLFW_KEY_A:
                moving_left = false;
                break;
            case GLFW_KEY_SPACE:
                moving_up = false;
                break;
            case GLFW_KEY_LEFT_SHIFT:
                moving_down = false;
                break;
            case GLFW_KEY_LEFT_CONTROL:
                fast = false;
                break;
        }
    }
}