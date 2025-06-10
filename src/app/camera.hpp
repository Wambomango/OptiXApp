#pragma once

#include "window.hpp"

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>


class Camera
{
    public:
        Camera(float fov, float aspect_ratio, float near_plane, float far_plane);
        void Tick(float dt);
        void AddCallbacks(Window& window);
        void SetPosition(const glm::vec3& position);
        void SetOrientation(const glm::vec3& orientation);
        glm::vec3 GetPosition();
        glm::vec3 GetOrientation();
        glm::mat4 GetViewMatrix();
        glm::mat4 GetProjectionMatrix();
        float GetFOV();
        float GetAspectRatio();
        float GetNearPlane();
        float GetFarPlane();
        void UpdateProjectionMatrix();
        void UpdateViewMatrix();
        
    private:    
        void OnScroll(double xoffset, double yoffset);
        void OnMouseMove(double xpos, double ypos);
        void OnMouseButton(int button, int action, int mods);
        void OnKey(int key, int scancode, int action, int mods);

        float fov;
        float aspect_ratio;
        float near_plane;
        float far_plane;
        glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f);
        glm::vec3 orientation = glm::vec3(0.0f, 0.0f, 0.0f);
        glm::mat4 view_matrix = glm::mat4(1.0f);
        glm::mat4 projection_matrix = glm::mat4(1.0f);


        bool moving_forward = false;
        bool moving_backward = false;
        bool moving_left = false;
        bool moving_right = false;
        bool moving_up = false;
        bool moving_down = false;
        bool fast = false;
};