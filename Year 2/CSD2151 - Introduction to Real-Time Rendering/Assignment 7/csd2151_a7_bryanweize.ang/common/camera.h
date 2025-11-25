/*!*****************************************************************************
\file camera.h
\author Vadim Surov (vsurov\@digipen.edu)
\par Course: CSD2151/CSD2150/CS250
\par Assignment: all
\date 12/26/2024 (MM/DD/YYYY)
\brief This file has definitions of the camera used in the framework 
       for scene definitions.
*******************************************************************************/
#pragma once

#include "program.h"

#include <glm/glm.hpp> // For mathematical types like vec2, vec3, mat4
#include <glm/ext.hpp> // For mathematical constants like pi
#include <glm/gtc/matrix_transform.hpp> // Utility functions for matrix transformations

namespace cg
{

    // Enumeration to define the type of camera
    enum CameraType { ORBITING, WALKING };

    //
    // Camera class to handle various camera functionalities
    //
    class Camera
    {
        CameraType cameraType; // Stores the type of camera (ORBITING or WALKING)
        glm::vec3 position; // Current position of the camera in 3D space
        glm::vec3 target;   // Target point that the camera is looking at
        float fieldOfView;  // Field of view angle in degrees
        float near;         // Near clipping plane
        float far;          // Far clipping plane

        double oldx; // Previous x-coordinate for cursor handling
        double oldy; // Previous y-coordinate for cursor handling

    public:
        // Constructor to initialize a camera with custom or default values
        Camera(
            CameraType cameraType,
            glm::vec3 position = { 0.0f, 3.0f, 3.0f },
            glm::vec3 target = { 0.0f, 0.0f, 0.0f },
            float fieldOfView = 45.0f,
            float near = 0.5f,
            float far = 200.0f
        ) :
            cameraType{ cameraType },
            position{ position },
            target{ target },
            fieldOfView{ fieldOfView },
            near{ near },
            far{ far },
            oldx{ -1.0 },
            oldy{ -1.0 }
        { }

        // Overloaded constructor with default values for an orbiting camera
        Camera(
            glm::vec3 position = { 0.0f, 3.0f, 3.0f },
            float fieldOfView = 45.0f
        ) :
            cameraType{ CameraType::ORBITING },
            position{ position },
            target{ 0.0f, 0.0f, 0.0f },
            fieldOfView{ fieldOfView },
            near{ 0.5f },
            far{ 200.0f },
            oldx{ -1.0 },
            oldy{ -1.0 }
        { }

        // Sets the camera type and updates the target if it's orbiting
        void setType(CameraType cameraType)
        {
            this->cameraType = cameraType;
            if (cameraType == CameraType::ORBITING)
                target = { 0.0f, 0.0f, 0.0f }; // Orbiting cameras always target the origin
        }

        // Updates the shader program with camera position
        void setUniforms(Program* pProgram) const
        {
            pProgram->use();
            pProgram->setUniform("camera.position", position); // Pass camera position to the shader
            pProgram->disuse();
        }

        // Returns the view matrix for the camera using the "look at" method
        glm::mat4 getLookAt(glm::vec3 center = { 0.0f, 0.0f, 0.0f }, glm::vec3 up = { 0.0f, 1.0f, 0.0f }) const
        {
            return glm::lookAt(position, target, up);
        }

        // Returns the perspective projection matrix based on the field of view and aspect ratio
        glm::mat4 getPerspective(float aspect = 1) const
        {
            return glm::perspective(glm::radians(fieldOfView), aspect, near, far);
        }

        // Handles cursor movement to adjust camera orientation
        void onCursor(double xoffset, double yoffset, Program* pProgram = nullptr)
        {
            const float PI05 = glm::pi<float>() / 2.0f; // Half of pi for clamping

            if (cameraType == CameraType::ORBITING)
            {
                // Calculate spherical coordinates for orbiting movement
                const float r = glm::sqrt(position.x * position.x +
                    position.y * position.y + position.z * position.z);
                float alpha = glm::asin(position.y / r); // Vertical angle
                float betta = std::atan2f(position.x, position.z); // Horizontal angle

                // Adjust angles based on cursor offset
                if (yoffset < 0.0)
                    alpha += -0.02f;
                else if (yoffset > 0.0)
                    alpha += 0.02f;

                if (xoffset < 0.0)
                    betta += 0.05f;
                else if (xoffset > 0.0)
                    betta += -0.05f;

                // Clamp vertical angle
                alpha = glm::clamp(alpha, -PI05 + 0.01f, PI05 - 0.01f);

                // Update position based on spherical coordinates
                position.x = r * glm::cos(alpha) * glm::sin(betta);
                position.y = r * glm::sin(alpha);
                position.z = r * glm::cos(alpha) * glm::cos(betta);
            }
            else if (cameraType == CameraType::WALKING)
            {
                // Calculate spherical coordinates for walking movement
                const float r = glm::sqrt(
                    (target.x - position.x) * (target.x - position.x) +
                    (target.y - position.y) * (target.y - position.y) + 
                    (target.z - position.z) * (target.z - position.z));
                float alpha = glm::asin((target.y - position.y) / r);
                float betta = std::atan2f((target.x - position.x), (target.z - position.z));

                // Adjust angles based on cursor offset
                if (yoffset < 0.0)
                    alpha += -0.02f;
                else if (yoffset > 0.0)
                    alpha += 0.02f;

                if (xoffset < 0.0)
                    betta += -0.05f;
                else if (xoffset > 0.0)
                    betta += 0.05f;

                // Clamp vertical angle
                alpha = glm::clamp(alpha, -PI05 + 0.01f, PI05 - 0.01f);

                // Update target based on spherical coordinates
                target.x = position.x + r * glm::cos(alpha) * glm::sin(betta);
                target.y = position.y + r * glm::sin(alpha);
                target.z = position.z + r * glm::cos(alpha) * glm::cos(betta);
            }

            // Update shader program with the new camera settings
            setUniforms(pProgram);
        }

        // Handles scroll input to adjust zoom or camera position
        void onScroll(double xoffset, double yoffset, Program* pProgram = nullptr)
        {
            if (cameraType == CameraType::ORBITING)
            {
                // Calculate the distance from the origin and adjust it based on scroll input
                float r = glm::sqrt(position.x * position.x + position.y * position.y + position.z * position.z);
                const float alpha = glm::asin(position.y / r);
                const float betta = std::atan2f(position.x, position.z);

                r += yoffset > 0.0f ? -1.0f : 1.0f; // Zoom in or out
                if (r < 1.0f) r = 1.0f; // Clamp minimum distance

                // Update position based on new distance
                position.x = r * glm::cos(alpha) * glm::sin(betta);
                position.y = r * glm::sin(alpha);
                position.z = r * glm::cos(alpha) * glm::cos(betta);
            }
            else if (cameraType == CameraType::WALKING)
            {
                // Adjust position and target based on scroll input for walking
                const glm::vec3 velocity = glm::normalize(yoffset > 0.0f ? target - position : position - target);
                position += velocity * glm::vec3(1.0f, 0.0f, 1.0f);
                target += velocity * glm::vec3(1.0f, 0.0f, 1.0f);
            }

            // Update shader program with the new camera settings
            setUniforms(pProgram);
        }
    };
}
