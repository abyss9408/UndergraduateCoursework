/*!*****************************************************************************
\file unittests_functions.cpp
\author Vadim Surov (vsurov\@digipen.edu)
\co-author YOUR NAME (DIGIPEN ID)
\par Course: CSD2151/CSD2150/CS250
\par Assignment: 5.1 (Environment Mapping)
\date 02/13/2022 (MM/DD/YYYY)
\brief This file has definitions of functions for unit tests.

This code is intended to be completed and submitted by a student,
so any changes in this file will be used in the evaluation on the VPL server.
You should not change functions' name and parameter types in provided code.
*******************************************************************************/

#include "unittests_functions.h"


template<typename T>
T floor(T a) { return glm::floor(a); }

template<typename T>
T mod(T a, T b) { return glm::mod(a, b); }

template<typename T>
T abs(T a) { return glm::abs(a); }

#define UNUSED(x) (void)x;

/*
    Read specs for Assignment 5.1
*/
vec4 checkerboardTexture(vec2 uv, float size)
{
    // Scale UV coordinates by size
    float x = floor(uv.x * size);
    float y = floor(uv.y * size);

    // Determine if we're in a black or white square
    bool isBlack = (static_cast<int>(x + y) & 1) == 0;

    return isBlack ? vec4(0.0f, 0.0f, 0.0f, 1.0f) : vec4(1.0f, 1.0f, 1.0f, 1.0f);
}

/*
    Read specs for Assignment 5.1
*/
vec2 vec2uv(vec3 v)
{
    float absX = abs(v.x);
    float absY = abs(v.y);
    float absZ = abs(v.z);

    // Check for top/bottom faces first
    if (absY > absX && absY > absZ || (v.x == v.y && v.x == v.z))
    {
        return vec2(0.0f, 0.0f);
    }

    if (absX > absZ || (absX == absY && absX == absZ)) // Left/Right faces
    {
        float scale = 1.0f / absX;
        if (v.x > 0.0f) // Right face
            return vec2(v.z * scale * 0.5f + 0.5f, v.y * scale * 0.5f + 0.5f);
        else // Left face
            return vec2(-v.z * scale * 0.5f + 0.5f, v.y * scale * 0.5f + 0.5f);
    }
    else // Front/Back faces
    {
        float scale = 1.0f / absZ;
        if (v.z > 0.0f) // Back face
            return vec2(-v.x * scale * 0.5f + 0.5f, v.y * scale * 0.5f + 0.5f);
        else // Front face
            return vec2(v.x * scale * 0.5f + 0.5f, v.y * scale * 0.5f + 0.5f);
    }
}