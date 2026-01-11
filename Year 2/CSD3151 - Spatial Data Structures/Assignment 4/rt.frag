#version 430

uniform sampler2D image;
out vec4 FragColor;

void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    vec4 v = texelFetch(image, pixel, 0);
    FragColor = v;
}
