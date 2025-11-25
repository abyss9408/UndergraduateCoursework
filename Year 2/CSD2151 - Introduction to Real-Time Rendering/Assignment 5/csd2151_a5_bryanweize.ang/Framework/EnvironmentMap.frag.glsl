/*!*****************************************************************************
\file EnvironmentMap.frag.glsl
\author Bryan Ang Wei Ze (bryanweize.ang\@digipen.edu)
\par Course: CSD2151
\par Assignment: 5.2
\date 02/09/2025 (MM/DD/YYYY)
\brief This file is the fragment shader for EnvironmentMap
*******************************************************************************/
R"(
#version 460 core

// Material properties for reflection and refraction
struct Material 
{
    vec4 color;             
    float reflectionFactor; // The light reflection factor
    float eta;              // The ratio of indices of refraction
};

uniform int Pass; // Pass number
uniform Material material;

layout(location=0) out vec4 FragColor;

// Pass 0 inputs
in vec3 Vec;

// Pass 1 inputs
in vec3 ReflectDir;
in vec3 RefractDir;

vec4 checkerboardTexture(vec2 uv, float size)
{
	// Scale UV coordinates by size
	float x = floor(uv.x * size);
	float y = floor(uv.y * size);

	// Determine if we're in a black or white square
	bool isBlack = (int(x + y) & 1) == 0;

	return isBlack ? vec4(0.0f, 0.0f, 0.0f, 1.0f) : vec4(1.0f, 1.0f, 1.0f, 1.0f);
}

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

void pass0() 
{
    vec3 color = checkerboardTexture(vec2uv(normalize(Vec)), 10.0f).rgb;
    color = pow(color, vec3(1.0f/2.2f)); // Gamma correction
    FragColor = vec4(color, 1.0f);
}

void pass1() 
{
    vec3 reflectColor = checkerboardTexture(vec2uv(normalize(ReflectDir)), 10.0f).rgb;
    vec3 refractColor = vec3(0.0);
    
    if (RefractDir != vec3(0.0)) // Check if refraction occurred
    {
        refractColor = checkerboardTexture(vec2uv(normalize(RefractDir)), 10.0f).rgb;
    }
    
    // Mix reflection and refraction
    vec3 color = mix(refractColor, reflectColor, material.reflectionFactor);
    
    // Apply material color
    color *= material.color.rgb;
    
    // Gamma correction
    color = pow(color, vec3(1.0f/2.2f));
    
    FragColor = vec4(color, 1.0f);
}

void main() 
{
	if      (Pass==0) pass0();
    else if (Pass==1) pass1();
}
)"