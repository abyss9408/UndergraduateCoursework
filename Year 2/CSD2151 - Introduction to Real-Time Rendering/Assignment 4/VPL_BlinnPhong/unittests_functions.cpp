/*!*****************************************************************************
\file unittests_functions.cpp
\author Vadim Surov (vsurov\@digipen.edu)
\co-author YOUR NAME (DIGIPEN ID)
\par Course: CSD2151/CSD2150/CS250
\par Assignment: 4.1 (BlinnPhong)
\date 02/06/2022 (MM/DD/YYYY)
\brief This file has definitions of functions for unit tests.

This code is intended to be completed and submitted by a student,
so any changes in this file will be used in the evaluation on the VPL server.
You should not change functions' name and parameter types in provided code.
*******************************************************************************/

#include "unittests_functions.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/extended_min_max.hpp>

template<typename T>
T min(T a, T b) { return glm::min(a, b); }

template<typename T>
T max(T a, T b) { return glm::max(a, b); }

template<typename T>
T abs(T a) { return glm::abs(a); }

template<typename T>
float length(T a) { return glm::length(a); }

template<typename T>
float dot(T a, T b) { return glm::dot(a, b); }

template<typename T>
T pow(T a, T b) { return glm::pow(a, b); }

#define UNUSED(x) (void)x;

/*
   Read specs for Assignment 4.1
*/
vec3 BlinnPhong(vec3 position, vec3 normal, Light light, Material material, mat4 view)
{
    // Transform light position to view space
    vec4 lightPosView = view * vec4(light.position, 1.0f);
    vec3 lightPos = vec3(lightPosView.x, lightPosView.y, lightPosView.z);

    // Calculate light vector (from fragment to light) in view space
    vec3 L = lightPos - position;
    if (length(L) < 0.0001f)
    {
        return vec3(0.0f);
    }
	L = normalize(L);

    // View vector in view space (from fragment to camera)
    // In view space, the camera is at (0,0,0)
    vec3 V = normalize(-position);

    // Calculate halfway vector
    vec3 H = normalize(L + V);

    // Ensure normal is normalized
    normal = normalize(normal);

    // Calculate the three components of Blinn-Phong reflection model

    // 1. Ambient component
    vec3 ambient = light.La * material.Ka;

    // 2. Diffuse component
    float diff = max(dot(normal, L), 0.0f);
    vec3 diffuse = light.Ld * material.Kd * diff;

    // 3. Specular component
    float spec = 0.0f;
    // Ensure light is actually hitting the surface
    if (diff > 0.0f)
    {
        spec = pow(max(dot(normal, H), 0.0f), material.shininess);
    }
    vec3 specular = light.Ls * material.Ks * spec;

    // Combine all components
    return ambient + diffuse + specular;
}