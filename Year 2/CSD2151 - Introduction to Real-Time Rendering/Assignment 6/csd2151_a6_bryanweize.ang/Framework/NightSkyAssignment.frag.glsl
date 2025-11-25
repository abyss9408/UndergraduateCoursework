/*!*****************************************************************************
\file NightSkyAssignment.frag.glsl
\author Bryan Ang Wei Ze (bryanweize.ang\@digipen.edu)
\par Course: CSD2151
\par Assignment: 8
\date 03/02/2025 (MM/DD/YYYY)
\brief This file is the fragment shader for NightSkyAssignment
*******************************************************************************/
R"(
#version 460 core

uniform int Pass; // Pass number

const vec3 FOG_COLOR = vec3(0.0f, 0.0f, 0.0f);
const float FOG_MAXDIST = 15.0f;
const float FOG_MINDIST = 5.0f;

const int CARTOON_LEVELS = 3;

const int CHECKERBOARD_SIZE = 50;
const float CHECKERBOARD_MIXLEVEL = 0.5f;

const int DISCARD_SCALE = 10;  // How many lines per polygon

struct Light
{
    vec3 position;
    vec3 L;   // Intensity for PBR
};

struct Material
{
    float rough, metal;
    vec3 color;
    float effect;
};

layout(location=0) out vec4 FragColor;

// Pass 0 (Skybox rendering)
layout(binding=0) uniform samplerCube CubeMapTex;
in vec3 Vec;

void pass0()
{
    vec3 color = texture(CubeMapTex, normalize(Vec)).rgb;
    FragColor = vec4(pow(color, vec3(1.0 / 2.2)), 1.0); // Gamma correction
}

// Pass 1 (Objects Rendering)
uniform mat4 V; 
uniform Material material;
uniform Light light[1];

in vec3 Position, Normal;
in vec2 TexCoord;

// Utility Functions
float cartoonShading(float value)
{
    return floor(value * CARTOON_LEVELS) / CARTOON_LEVELS;
}

vec4 checkerboardTexture(vec2 uv, float size)
{
    float checker = mod(floor(uv.x * size) + floor(uv.y * size), 2.0);
    return vec4(checker, checker, checker, 1.0);
}

// Hash function for noise-based discard
float hash(vec2 p)
{
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

// Microfacet Model (PBR)
const float PI = 3.14159265358979323846;

float ggxDistribution(float nDotH)
{
    float alpha2 = material.rough * material.rough * material.rough * material.rough;
    float d = (nDotH * nDotH) * (alpha2 - 1.0f) + 1.0f;
    return alpha2 / (PI * d * d);
}

float geomSmith(float nDotL)
{
    float k = (material.rough + 1.0) * (material.rough + 1.0) / 8.0;
    return 1.0 / (nDotL * (1.0 - k) + k);
}

vec3 schlickFresnel(float lDotH)
{
    vec3 f0 = vec3(0.04);
    if (material.metal == 1.0) f0 = material.color;
    return f0 + (1.0 - f0) * pow(1.0 - lDotH, 5);
}

vec3 microfacetModel(vec3 position, vec3 n, bool isCartoon)
{  
    vec3 lightI = light[0].L;
    vec3 lightPosView = (V * vec4(light[0].position, 1.0)).xyz;
    
    vec3 l = normalize(lightPosView - position);
    vec3 v = normalize(-position);
    vec3 h = normalize(v + l);

    float nDotL = max(dot(n, l), 0.0);
    float nDotH = dot(n, h);
    float lDotH = dot(l, h);
    float nDotV = dot(n, v);

    if (isCartoon)
{
        nDotL = cartoonShading(nDotL);
        nDotH = cartoonShading(nDotH);
        lDotH = cartoonShading(lDotH);
        nDotV = cartoonShading(nDotV);
    }

    vec3 specBrdf = 0.25 * ggxDistribution(nDotH) * schlickFresnel(lDotH) * geomSmith(nDotL) * geomSmith(nDotV);
    return (material.color + PI * specBrdf) * lightI * nDotL;
}

// Discard Effects
bool applyDiscardEffect()
{
    vec2 f = fract(TexCoord * DISCARD_SCALE);
    return hash(f) < 0.3;  // 30% chance to discard
}

// Pass 1 Main Function
void pass1()
{
    bool isCartoon = (material.effect == 2.0);

    vec4 frontColor = vec4(microfacetModel(Position, normalize(Normal), isCartoon), 1.0);
    vec4 backColor = vec4(microfacetModel(Position, normalize(-Normal), false), 1.0);
    vec4 effectColor;

    if (material.effect == 0.0) 
    {
        // No effect
        effectColor = frontColor;
    }
    else if (material.effect == 1.0)
    {
        // Discard
        if (applyDiscardEffect()) discard;
        effectColor = gl_FrontFacing ? frontColor : backColor;
    } 
    else if (material.effect == 2.0)
    {
        // Cartoon
        effectColor = frontColor;
    } 
    else if (material.effect == 3.0)
    {   // Checkerboard
        effectColor = mix(frontColor, checkerboardTexture(TexCoord, CHECKERBOARD_SIZE), CHECKERBOARD_MIXLEVEL);
    } 
    else
    {
        discard;
    }

    // Apply Fog
    float fogFactor = clamp((FOG_MAXDIST - length(Position)) / (FOG_MAXDIST - FOG_MINDIST), 0.0, 1.0);
    vec3 color = mix(FOG_COLOR, effectColor.rgb, fogFactor);

    // Gamma correction
    FragColor = vec4(pow(color, vec3(1.0 / 2.2)), 1.0);
}

void main() {
    if      (Pass==0) pass0();
    else if (Pass==1) pass1();
}
)"