/*!*****************************************************************************
\file FiltersAssignment.frag.glsl
\author Bryan Ang Wei Ze (bryanweize.ang\@digipen.edu)
\par Course: CSD2151
\par Assignment: 9
\date 03/09/2025 (MM/DD/YYYY)
\brief This file is the fragment shader for FiltersAssignment
*******************************************************************************/
R"(
#version 460 core

uniform int Pass;       // Pass number
layout(location=0) out vec4 FragColor;

// Lighting Structures
struct Light
{
	vec3 position;      // Position of the light source in world space
    vec3 La;            // Ambient light intensity
    vec3 Ld;            // Diffuse light intensity
    vec3 Ls;            // Specular light intensity
};

struct Material
{
    vec3 Ka;            // Ambient reflectivity
    vec3 Kd;            // Diffuse reflectivity
    vec3 Ks;            // Specular reflectivity
    float shininess;    // Specular shininess factor
};

// Pass 0: Blinn-Phong Shading
in vec3 Position;
in vec3 Normal;

uniform Light light[1];
uniform Material material;
uniform mat4 V;         // View transform matrix

vec3 blinnPhong(vec3 position, vec3 normal, int lightIndex)
{
    // Ambient component
    vec3 ambient = light[lightIndex].La * material.Ka;

    // Transform light position to view space
    vec3 lightPositionInView = (V * vec4(light[lightIndex].position, 1.0f)).xyz;
    vec3 s = normalize(lightPositionInView - position);

    // Diffuse component
    float sDotN = max(dot(s, normal), 0.0f);
    vec3 diffuse = light[lightIndex].Ld * material.Kd * sDotN;

    // Specular component
    vec3 spec = vec3(0.0f);
    if (sDotN > 0.0f)
    {
        vec3 v = normalize(-position.xyz);
        vec3 h = normalize(v + s);
        spec = light[lightIndex].Ls * material.Ks * 
            pow(max(dot(h, normal), 0.0f), material.shininess);
    }
    
    return ambient + diffuse + spec;
}

// Pass 1 & 2: Gaussian Blur
layout(binding=0) uniform sampler2D Texture;

// Precomputed normalized Gaussian weights
const float GaussianWeights[5] = { 0.158435f, 0.148836f, 0.123389f, 0.0902733f, 0.0582848f };

// Apply Gaussian blur in a specified direction
vec4 applyGaussianBlur(ivec2 pixelCoord, bool isVertical)
{
    vec4 sum = texelFetch(Texture, pixelCoord, 0) * GaussianWeights[0];
    
    for (int i = 1; i < 5; i++)
    {
        ivec2 offset1 = isVertical ? ivec2(0, -i) : ivec2(-i, 0);
        ivec2 offset2 = isVertical ? ivec2(0, i) : ivec2(i, 0);
        
        sum += texelFetchOffset(Texture, pixelCoord, 0, offset1) * GaussianWeights[i];
        sum += texelFetchOffset(Texture, pixelCoord, 0, offset2) * GaussianWeights[i];
    }
    
    return sum;
}

// Pass 3: Sobel Edge Detection
uniform float EdgeThreshold = 0.03f;

float luminance(vec3 color)
{
    return dot(vec3(0.2126f, 0.7152f, 0.0722f), color);
}

vec4 detectEdges(ivec2 pixelCoord)
{
    // Sample neighboring pixels
    float s00 = luminance(texelFetchOffset(Texture, pixelCoord, 0, ivec2(-1, 1)).rgb);
    float s10 = luminance(texelFetchOffset(Texture, pixelCoord, 0, ivec2(-1, 0)).rgb);
    float s20 = luminance(texelFetchOffset(Texture, pixelCoord, 0, ivec2(-1,-1)).rgb);
    float s01 = luminance(texelFetchOffset(Texture, pixelCoord, 0, ivec2( 0, 1)).rgb);
    float s21 = luminance(texelFetchOffset(Texture, pixelCoord, 0, ivec2( 0,-1)).rgb);
    float s02 = luminance(texelFetchOffset(Texture, pixelCoord, 0, ivec2( 1, 1)).rgb);
    float s12 = luminance(texelFetchOffset(Texture, pixelCoord, 0, ivec2( 1, 0)).rgb);
    float s22 = luminance(texelFetchOffset(Texture, pixelCoord, 0, ivec2( 1,-1)).rgb);

    // Sobel operator for edge detection
    float sx = s00 + 2*s10 + s20 - (s02 + 2*s12 + s22);
    float sy = s00 + 2*s01 + s02 - (s20 + 2*s21 + s22);

    float gradient = sx*sx + sy*sy;
    
    return (gradient > EdgeThreshold) ? vec4(1.0f) : vec4(0.0f, 0.0f, 0.0f, 1.0f);
}

// Pass implementation functions
void pass0()
{
    FragColor = vec4(blinnPhong(Position, normalize(Normal), 0), 1.0f);
}

void pass1()
{
    ivec2 pixelCoord = ivec2(gl_FragCoord.xy);
    FragColor = applyGaussianBlur(pixelCoord, true); // Vertical blur
}

void pass2()
{
    ivec2 pixelCoord = ivec2(gl_FragCoord.xy);
    FragColor = applyGaussianBlur(pixelCoord, false); // Horizontal blur
}

void pass3()
{
    ivec2 pixelCoord = ivec2(gl_FragCoord.xy);
    FragColor = detectEdges(pixelCoord);
}

void main()
{
    switch (Pass)
    {
        case 0: pass0(); break;
        case 1: pass1(); break;
        case 2: pass2(); break;
        case 3: pass3(); break;
    }
}
)"