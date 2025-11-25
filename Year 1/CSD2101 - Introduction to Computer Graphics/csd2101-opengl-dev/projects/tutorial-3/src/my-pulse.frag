R"( #version 450 core
	
	layout (location=0) in vec3 vInterpolatedColor;
	layout (location=0) out vec4 fFragColor;

	uniform float uTime;
	
	void main()
	{
		fFragColor = vec4(vInterpolatedColor.r, vInterpolatedColor.g * (sin(uTime * 2.0) + 1.0) / 2.0, vInterpolatedColor.b, 1.0);
	}
)"