#pragma once

#include "window.hpp"

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>


class Camera
{
    public:
        Camera(float fov, float aspect_ratio, float near_plane, float far_plane) : fov(fov), aspect_ratio(aspect_ratio), near_plane(near_plane), far_plane(far_plane)
        {
            UpdateProjectionMatrix();
        }

        void SetCallbacks(Window& window)
        {
            window.SetScrollCallback(std::bind(&Camera::OnScroll, this, std::placeholders::_1, std::placeholders::_2));
            window.SetMouseMoveCallback(std::bind(&Camera::OnMouseMove, this, std::placeholders::_1, std::placeholders::_2));
            window.SetMouseButtonCallback(std::bind(&Camera::OnMouseButton, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
            window.SetKeyCallback(std::bind(&Camera::OnKey, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
        }

        void SetPosition(const glm::vec3& position)
        {
            this->position = position;
            UpdateViewMatrix();
        }
        void SetOrientation(const glm::vec3& orientation)
        {
            this->orientation = orientation;
            UpdateViewMatrix();
        }
        glm::mat4 GetViewMatrix() const
        {
            return view_matrix;
        }
        glm::mat4 GetProjectionMatrix() const
        {
            return projection_matrix;
        }
        void UpdateProjectionMatrix()
        {
            projection_matrix = glm::perspective(glm::radians(fov), aspect_ratio, near_plane, far_plane);
        }
        void UpdateViewMatrix()
        {
            view_matrix = glm::eulerAngleX(glm::radians(-orientation.y)) * glm::eulerAngleY(glm::radians(orientation.x)) * glm::translate(glm::mat4(1.0f), -position);
        }
        
    private:    

        void OnScroll(double xoffset, double yoffset)
        {
            fov -= static_cast<float>(yoffset);
            fov = glm::clamp(fov, 1.0f, 45.0f); 
            UpdateProjectionMatrix();
        }

        void OnMouseMove(double xpos, double ypos)
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

        void OnMouseButton(int button, int action, int mods)
        {
        }
    
        void OnKey(int key, int scancode, int action, int mods)
        {
            float sensitivity = 0.1f; 
            
            glm::vec3 forward = glm::normalize(glm::vec3(
                glm::sin(glm::radians(orientation.x)),
                0.0f,
                -glm::cos(glm::radians(orientation.x))
            ));
            glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));
            glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);

            if (action == GLFW_PRESS || action == GLFW_REPEAT)
            {
                switch (key)
                {
                    case GLFW_KEY_W:
                        position += sensitivity * forward ;
                        break;
                    case GLFW_KEY_S:
                        position -= sensitivity * forward;
                        break;
                    case GLFW_KEY_D:
                        position += sensitivity * right;
                        break;
                    case GLFW_KEY_A:
                        position -= sensitivity * right;     
                        break;
                    case GLFW_KEY_E:
                        position += sensitivity * up;
                        break;
                    case GLFW_KEY_Q:
                        position -= sensitivity * up;
                        break;
                }

                UpdateViewMatrix();
            }
        }

        float fov;
        float aspect_ratio;
        float near_plane;
        float far_plane;
        glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f);
        glm::vec3 orientation = glm::vec3(0.0f, 0.0f, 0.0f);
        glm::mat4 view_matrix = glm::mat4(1.0f);
        glm::mat4 projection_matrix = glm::mat4(1.0f);
};