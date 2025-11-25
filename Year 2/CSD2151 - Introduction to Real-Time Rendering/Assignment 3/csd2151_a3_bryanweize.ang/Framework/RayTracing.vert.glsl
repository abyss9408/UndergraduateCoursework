R"(
#version 330 core

layout(location=0) in vec3 VertexPosition;
layout(location=1) in vec3 VertexNormal;
layout(location=2) in vec3 VertexTexCoord;

out vec3 TexCoord;

void main()
{
	TexCoord = VertexTexCoord;
    gl_Position = vec4(VertexPosition.x, VertexPosition.y, 0.0f, 1.0f);
}
)"