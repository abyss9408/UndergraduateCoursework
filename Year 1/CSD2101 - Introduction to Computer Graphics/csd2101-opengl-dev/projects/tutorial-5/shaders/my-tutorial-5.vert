#version 450 core

layout (location=0) in vec2 aVertexPosition;
layout (location=1) in vec3 aVertexColor;
layout (location=2) in vec2 aVertexTexture;
	
layout (location=0) out vec3 vColor;
layout (location=1) out vec2 vTexCoord;

uniform int uTask;
uniform mat2 uRotMtx;
uniform bool uRotate;
uniform vec2 uMcn;

void main()
{
	gl_Position = vec4(aVertexPosition, 0.0, 1.0);
	vColor = aVertexColor;

	if (uRotate)
	{	
		if (uTask == 4 || uTask == 5 || uTask == 6)
		{
			vTexCoord = uRotMtx * 4 * (aVertexTexture - uMcn);
		}
		else
		{
			vTexCoord = uRotMtx * (aVertexTexture - uMcn);
		}
	}
	else
	{
		if (uTask == 4 || uTask == 5 || uTask == 6)
		{
			vTexCoord = 4 * aVertexTexture;
		}
		else
		{
			vTexCoord = aVertexTexture;
		}
	}
}