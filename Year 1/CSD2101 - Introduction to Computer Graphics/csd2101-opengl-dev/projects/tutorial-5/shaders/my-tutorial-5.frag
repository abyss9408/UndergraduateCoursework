#version 450 core
	
layout (location=0) in vec3 vInterpolatedColor;
layout (location=1) in vec2 vTexCoord;
layout (location=0) out vec4 fFragColor;

uniform int uTask;
uniform float uTileSize;
uniform sampler2D uTex2d;
uniform bool uModulate;
	
void main()
{
	// modulation mode
	if (uModulate)
	{
		switch (uTask)
		{
		case 0:
			fFragColor = vec4(vInterpolatedColor, 1.0);
			break;
		case 1:
			if (mod(floor(gl_FragCoord.x / 32), 2) == 0 && mod(floor(gl_FragCoord.y / 32), 2) == 0 || 
			mod(floor(gl_FragCoord.x / 32), 2) == 1 && mod(floor(gl_FragCoord.y / 32), 2) == 1)
			{
				fFragColor = vec4(1.0, 0.0, 1.0, 1.0) * vec4(vInterpolatedColor, 1.0);
			}
			else
			{
				fFragColor = vec4(0.0, 0.68, 0.94, 1.0) * vec4(vInterpolatedColor, 1.0);
			}
			break;
		case 2:
			if (mod(floor(gl_FragCoord.x / uTileSize), 2) == 0 && mod(floor(gl_FragCoord.y / uTileSize), 2) == 0 || 
			mod(floor(gl_FragCoord.x / uTileSize), 2) == 1 && mod(floor(gl_FragCoord.y / uTileSize), 2) == 1)
			{
				fFragColor = vec4(1.0, 0.0, 1.0, 1.0) * vec4(vInterpolatedColor, 1.0);
			}
			else
			{
				fFragColor = vec4(0.0, 0.68, 0.94, 1.0) * vec4(vInterpolatedColor, 1.0);
			}
			break;
		default:
			fFragColor = texture(uTex2d, vTexCoord) * vec4(vInterpolatedColor, 1.0);
		}
	}
	// default mode
	else
	{
		switch (uTask)
		{
		case 0:
			fFragColor = vec4(vInterpolatedColor, 1.0);
			break;
		case 1:
			if (mod(floor(gl_FragCoord.x / 32), 2) == 0 && mod(floor(gl_FragCoord.y / 32), 2) == 0 || 
			mod(floor(gl_FragCoord.x / 32), 2) == 1 && mod(floor(gl_FragCoord.y / 32), 2) == 1)
			{
				fFragColor = vec4(1.0, 0.0, 1.0, 1.0);
			}
			else
			{
				fFragColor = vec4(0.0, 0.68, 0.94, 1.0);
			}
			break;
		case 2:
			if (mod(floor(gl_FragCoord.x / uTileSize), 2) == 0 && mod(floor(gl_FragCoord.y / uTileSize), 2) == 0 || 
			mod(floor(gl_FragCoord.x / uTileSize), 2) == 1 && mod(floor(gl_FragCoord.y / uTileSize), 2) == 1)
			{
				fFragColor = vec4(1.0, 0.0, 1.0, 1.0);
			}
			else
			{
				fFragColor = vec4(0.0, 0.68, 0.94, 1.0);
			}
			break;
		default:
			fFragColor = texture(uTex2d, vTexCoord);
		}
	}
}