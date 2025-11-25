R"( #version 450 core
	
	layout (location=0) in vec3 vInterpolatedColor;
	layout (location=0) out vec4 fFragColor;
	
	void main()
	{
		fFragColor = vec4(0.0, 0.0, 0.0, 1.0);
	}
)"