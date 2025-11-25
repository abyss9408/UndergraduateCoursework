R"(
#version 330 core

in vec3 TexCoord;

uniform vec2 iResolution;
uniform float iTime;
uniform bool resetState;

layout(location=0) out vec4 FragColor;

struct Light
{
	vec3 position;
};

struct Ray
{
	vec3 origin;
	vec3 direction;
};

struct Sphere
{
	vec3 center;
	float radius;
};

float time_intersect_ray_sphere(Ray ray, Sphere sphere)
{
	float b = dot((ray.origin - sphere.center) + (ray.origin - sphere.center), ray.direction);
	float c =dot((ray.origin - sphere.center), (ray.origin - sphere.center)) - sphere.radius * sphere.radius;
    float intersection = 0.0f;

	if (4 * c > b * b)
    {
        return intersection;
    }
	else if (4 * c == b * b)
	{
		intersection = -b / 2;
	}
	else
	{
		float sqrtValue = sqrt(b * b - 4 * c);
		float t1 = (-b + sqrtValue) / 2;
		float t2 = (-b - sqrtValue) / 2;

		if (t1 < 0 && t2 > 0)
		{
			intersection = t2;
		}
		else if (t2 < 0 && t1 > 0)
		{
			intersection = t1;
		}
		else if (t1 > 0 && t2 > 0)
		{
			intersection = min(t1, t2);
		}
	}

	return intersection;
}

vec3 rayTracing(Ray ray, Sphere sphere, Light light)
{
    float t = time_intersect_ray_sphere(ray, sphere);
	vec3 finalColor = vec3(0.1f);

	if (t > 0.0f)
	{
		vec3 intersectionPoint = ray.origin + t * ray.direction;
		vec3 normal = normalize(intersectionPoint - sphere.center);
		vec3 lightDir = normalize(light.position - intersectionPoint);
		float diff = max(0.1f, dot(normal, lightDir));

		// Dynamic color based on time
        vec3 color = vec3(sin(iTime) * 0.5 + 0.5, 0.2, 0.3);
		finalColor = color * diff;
	}
    return finalColor;
}

void main() 
{
	vec2 uv = (gl_FragCoord.xy * 2.0f - iResolution.xy) / iResolution.y;
    vec3 xyz = vec3(uv, -1.0f);
    vec3 direction = normalize(xyz); 
    Ray ray = Ray(vec3(0.0f), direction);

	// Dynamic position based on time
	vec3 spherePos = vec3(sin(iTime) * 0.5, cos(iTime) * 0.5, -2.5f);
    Sphere sphere = Sphere(resetState ? vec3(0.0f, 0.0f, -2.5f) : spherePos, 1.0f);
    Light light = Light(vec3(10.0f, 10.0f, 10.0f));

    FragColor = vec4(rayTracing(ray, sphere, light), 1.0f);
}
)"