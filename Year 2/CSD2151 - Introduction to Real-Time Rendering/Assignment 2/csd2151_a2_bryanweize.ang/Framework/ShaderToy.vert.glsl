R"(
#version 330 core

/*
   2D Pass-Through Vertex Shader 
*/

layout(location=0) in vec3 VertexPosition;
layout(location=1) in vec3 VertexNormal;
layout(location=2) in vec3 VertexTexCoord;

void main()
{
	gl_Position = vec4(VertexPosition.x, VertexPosition.y, 0.0f, 1.0f);
}
)"