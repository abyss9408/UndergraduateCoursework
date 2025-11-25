R"(
#version 330 core

//
// Inspired by https://www.shadertoy.com/view/mtyGWy
// All credits to kishimisu: https://www.shadertoy.com/user/kishimisu
//

uniform vec2 iResolution;   // Viewport resolution (in pixels)
uniform float iTime;        // Shader playback time (in seconds)

layout(location=0) out vec4 FragColor;

// Mathemical function: Enhanced palette with trigonometric functions
vec3 palette(float t)
{
    // Dynamic color coefficients that change with time
    vec3 a = vec3(0.5f + 0.2f * sin(iTime * 0.5f));
    vec3 b = vec3(0.5f + 0.2f * cos(iTime * 0.3f));
    vec3 c = vec3(1.0f + 0.3f * sin(iTime * 0.2f));
    vec3 d = vec3(0.263f, 0.416f, 0.557f) + 0.1f * cos(iTime * 0.4f);

    return a + b * cos(6.28318f * (c * t + d));
}

// Complex geomtric shape: Mandelbulb-inspired function
float geometricShape(vec2 uv, float time)
{
    vec2 z = uv;
    // Power for the shape
    float n = 8.0f;
    float r = 0.0f;

    for (int i = 0; i < 3; ++i)
    {
        float theta = atan(z.y, z.x) * n;
        float r = length(z);
        z = vec2(cos(theta), sin(theta)) * pow(r, n) + uv;
        if (r > 2.0f)
        {
            break;
        }
    }

    return 1.0f / (1.0f + length(z) * exp(-sin(time)));
}

void main()
{
    // Normalized coordinates to cover viewport
    vec2 uv = (gl_FragCoord.xy * 2.0f - iResolution.xy) / iResolution.y;
    float luv = length(uv);

    // Init final color
    vec3 finalColor = vec3(0.0f);

    // Element 1: Mathemical function visualization (geometric shape)
    float shape = geometricShape(uv * 0.5f, iTime);

    // Element 2: Dynamic color variations (changes with time)
    vec3 dynamicColor = palette(luv + iTime * 0.4f);

    // Element 3: Dynamic position/size element
    vec2 movingUV = uv + vec2(sin(iTime) * 0.5f, cos(iTime * 0.7f) * 0.3f);
    float movingCircle = 1.0f / (1.0f + 30.0f * length(movingUV));

    // Element 4: Colorful element with multiple shades
    for (int i = 0; i < 4; ++i) 
    {
        vec2 fracUV = fract(uv * (1.5f + float(i) * 0.2f)) - 0.5f;
        float d = pow(1.0f / (100.0f * abs(sin(length(fracUV) + iTime) / 8.0f)), 1.2f);
        vec3 col = palette(luv + i * 0.4f + iTime * 0.4f);
        finalColor += col * d * 0.3f;
    }

    // Element 5: Random element using noise-like pattern
    float random = fract(sin(dot(uv + iTime, vec2(12.9898f, 78.233f))) * 43758.5453123f);
    float randomPattern = random * 0.15f * (1.0f + sin(iTime));
    
    // Combine all elements
    finalColor += dynamicColor * shape * 0.5f;          // Mathematical and color elements
    finalColor += vec3(movingCircle) * 0.3f;           // Dynamic position element
    finalColor += vec3(randomPattern);                  // Random element
    
    // Ensure the shader covers the entire viewport
    finalColor = mix(finalColor, palette(luv), 0.1f);
    
    // Output final color
    FragColor = vec4(finalColor, 1.0f);
}
)"