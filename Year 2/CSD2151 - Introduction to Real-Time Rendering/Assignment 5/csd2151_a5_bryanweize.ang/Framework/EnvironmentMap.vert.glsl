/*!*****************************************************************************
\file EnvironmentMap.vert.glsl
\author Bryan Ang Wei Ze (bryanweize.ang\@digipen.edu)
\par Course: CSD2151
\par Assignment: 5.2
\date 02/09/2025 (MM/DD/YYYY)
\brief This file is the vertex shader for EnvironmentMap
*******************************************************************************/
R"(
#version 460 core

struct Camera 
{
    vec3 position;    // In world space
};

struct Material 
{
    vec4 color;             
    float reflectionFactor; // The light reflection factor
    float eta;              // The ratio of indices of refraction
};

uniform int Pass; // Pass number
uniform mat4 M; // Model transform matrix
uniform mat4 V; // View transform matrix
uniform mat4 P; // Projection transform matrix
uniform Camera camera;
uniform Material material;


layout(location=0) in vec3 VertexPosition;
layout(location=1) in vec3 VertexNormal;

// Pass 0
out vec3 Vec;


// Pass 1
out vec3 ReflectDir;
out vec3 RefractDir;

void pass0()
{
    Vec = VertexPosition; 
    gl_Position = P * V * M * vec4(VertexPosition, 1.0f);
}

void pass1()
{
    // Transform vertex position and normal to world space
    vec4 worldPos = M * vec4(VertexPosition, 1.0f);
    mat3 normalMatrix = transpose(inverse(mat3(M)));
    vec3 worldNormal = normalize(normalMatrix * VertexNormal);
    
    // Calculate view direction (from vertex to camera)
    vec3 viewDir = normalize(camera.position - worldPos.xyz);
    
    // Calculate reflection and refraction directions
    ReflectDir = reflect(-viewDir, worldNormal);
    RefractDir = refract(-viewDir, worldNormal, material.eta);
    
    gl_Position = P * V * worldPos;
}

void main()
{
	if      (Pass==0) pass0();
    else if (Pass==1) pass1();
}
)"