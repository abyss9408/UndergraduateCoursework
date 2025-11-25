/*!*****************************************************************************
\file BlinnPhong.frag.glsl
\author Vadim Surov (vsurov\@digipen.edu)
\co-author Bryan Ang Wei Ze (bryanweize.ang\@digipen.edu)
\par Course: CSD2151
\par Assignment: 4.2
\date 02/02/2025 (MM/DD/YYYY)
\brief This file is the fragment shader for the Blinn-Phong shading model.
*******************************************************************************/
R"(
#version 330 core

struct Material 
{
    vec3 Ka;            // Ambient reflectivity
    vec3 Kd;            // Diffuse reflectivity
    vec3 Ks;            // Specular reflectivity
    float shininess;    // Specular shininess factor
};

struct Light 
{
    vec3 position;      // Position of the light source in the world space
    vec3 La;            // Ambient light intensity
    vec3 Ld;            // Diffuse light intensity
    vec3 Ls;            // Specular light intensity
};

in vec3 Position;       // In view space
in vec3 Normal;         // In view space

uniform Light light[1];
uniform Material material;
uniform mat4 V;         // View transform matrix

layout(location=0) out vec4 FragColor;

vec3 BlinnPhong(vec3 position, vec3 normal, Light light, Material material, mat4 view)
{
    // Transform light position to view space
    vec4 lightPosView = view * vec4(light.position, 1.0f);
    vec3 lightPos = vec3(lightPosView.x, lightPosView.y, lightPosView.z);
    
    // Calculate light vector (from fragment to light) in view space
    vec3 L = lightPos - position;
    if (length(L) < 0.0001f)
    {
        return vec3(0.0f);
    }
    L = normalize(L);
    
    // View vector in view space (from fragment to camera)
    // In view space, the camera is at (0,0,0)
    vec3 V = normalize(-position);
    
    // Calculate halfway vector
    vec3 H = normalize(L + V);
    
    // Ensure normal is normalized
    normal = normalize(normal);
    
    // Calculate the three components of Blinn-Phong reflection model
    
    // 1. Ambient component
    vec3 ambient = light.La * material.Ka;
    
    // 2. Diffuse component
    float diff = max(dot(normal, L), 0.0f);
    vec3 diffuse = light.Ld * material.Kd * diff;
    
    // 3. Specular component
    float spec = 0.0f;
    // Ensure light is actually hitting the surface
    if (diff > 0.0f)
    {
        spec = pow(max(dot(normal, H), 0.0f), material.shininess);
    }
    vec3 specular = light.Ls * material.Ks * spec;
    
    // Combine all components
    return ambient + diffuse + specular;
}

void main() 
{
    FragColor = vec4(BlinnPhong(Position, normalize(Normal), light[0], material, V), 1.0f);
}
)"