/*!*****************************************************************************
\file BlinnPhong.vert.glsl
\author Vadim Surov (vsurov\@digipen.edu)
\co-author Bryan Ang Wei Ze (bryanweize.ang\@digipen.edu)
\par Course: CSD2151
\par Assignment: 4.2
\date 02/02/2025 (MM/DD/YYYY)
\brief This file is the vertex shader for the Blinn-Phong shading model.
*******************************************************************************/
R"(
#version 330 core

/*
    This vertex shader simply converts the position and normal 
    to camera space and passes them to the fragment shader.
*/

layout(location=0) in vec3 VertexPosition;
layout(location=1) in vec3 VertexNormal;

out vec3 Position;
out vec3 Normal;

uniform mat4 M; // Model transform matrix
uniform mat4 V; // View transform matrix
uniform mat4 P; // Projection transform matrix

void main()
{
	mat4 MV = V * M; // Model-View transform matrix

    mat3 N = mat3(vec3(MV[0]), vec3(MV[1]), vec3(MV[2])); // Normal transform matrix
    Normal = normalize(N * VertexNormal);

    vec4 VertexPositionInView = MV * vec4(VertexPosition, 1.0f);
    Position = VertexPositionInView.xyz;
    gl_Position = P * VertexPositionInView;
}
)"