/*!*****************************************************************************
\file NightSkyAssignment.vert.glsl
\author Bryan Ang Wei Ze (bryanweize.ang\@digipen.edu)
\par Course: CSD2151
\par Assignment: 8
\date 03/02/2025 (MM/DD/YYYY)
\brief This file is the vertex shader for NightSkyAssignment
*******************************************************************************/
R"(
#version 460 core

/*
   This vertex shader converts the position and normal to camera space 
   and passes them to the fragment shader, along with texture coordinates.
*/

uniform int Pass; // Pass number

layout(location=0) in vec3 VertexPosition;
layout(location=1) in vec3 VertexNormal;
layout(location=2) in vec2 VertexTexCoord;

uniform mat4 M; // Model transform matrix
uniform mat4 V; // View transform matrix
uniform mat4 P; // Projection transform matrix


// Pass 0

out vec3 Vec;

void pass0()
{
    Vec = VertexPosition; 
    mat4 VV = mat4(V[0],V[1],V[2],vec4(0,0,0,1)); // To do not move the Skybox with camera
    gl_Position = P * VV * M * vec4(VertexPosition, 1.0f);
}

// Pass 1

out vec3 Position;
out vec3 Normal;
out vec2 TexCoord;

void pass1()
{
    // Get the position and normal in view space
    mat4 MV = V * M;
    mat3 N = mat3(vec3(MV[0]), vec3(MV[1]), vec3(MV[2])); // Normal transform matrix
    vec3 VertexNormalInView = normalize(N * VertexNormal);
    vec4 VertexPositionInView = MV * vec4(VertexPosition, 1.0f);
    
    Position = VertexPositionInView.xyz;
    Normal = VertexNormalInView;
    TexCoord = VertexTexCoord;

    gl_Position = P * VertexPositionInView;
}

void main()
{
    if      (Pass==0) pass0();
    else if (Pass==1) pass1();
}
)"