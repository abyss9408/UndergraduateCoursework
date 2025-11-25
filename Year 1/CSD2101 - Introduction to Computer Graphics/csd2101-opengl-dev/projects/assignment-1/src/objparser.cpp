/*!
@file       objparser.cpp
@author     bryanweize.ang@digipen.edu
@date       10/07/2024

This file defines a function that processes .obj files

*//*__________________________________________________________________________*/

#include "objparser.h"

bool MyObjParser::parse_obj_mesh(std::string filename, std::vector<glm::vec3>& positions_normalized,
	std::vector<glm::vec3>& normals, std::vector<glm::vec2>& tex_coords,
	std::vector<unsigned short>& triangles, bool load_nml_coord_flag, bool load_tex_coord_flag,
	bool model_centered_flag)
{
	std::ifstream file(filename);

	if (!file.is_open())
	{
		return false;
	}

	// read in obj file data
	std::string line;
	std::vector<Face> faces;
	while (std::getline(file, line))
	{
		std::istringstream iss(line);
		std::string prefix;
		iss >> prefix;

		if (prefix == "v")
		{
			glm::vec3 position;
			iss >> position.x >> position.y >> position.z;
			positions_normalized.push_back(position);
		}
		else if (prefix == "vt" && load_tex_coord_flag)
		{
			glm::vec2 tex_coord;
			iss >> tex_coord.x >> tex_coord.y;
			tex_coords.push_back(tex_coord);
		}
		else if (prefix == "vn" && load_nml_coord_flag)
		{
			glm::vec3 normal;
			iss >> normal.x >> normal.y >> normal.z;
			normals.push_back(normal);
		}
		else if (prefix == "f")
		{
			Face face;
			for (size_t i{}; i < 3; ++i)
			{
				std::string indices_data;
				iss >> indices_data;
				std::replace(indices_data.begin(), indices_data.end(), '/', ' ');
				std::istringstream indices_stream(indices_data);
				indices_stream >> face.vertexIndices[i] >> face.textureIndices[i] >> face.normalIndices[i];
				--face.vertexIndices[i];
				--face.textureIndices[i];
				--face.normalIndices[i];
				triangles.push_back(face.vertexIndices[i]);
			}
			faces.push_back(face);
		}
	}

	file.close();

	// compute centeroid of model using mathematical center
	if (model_centered_flag)
	{
		glm::vec3 min_pos{ positions_normalized[0] };
		glm::vec3 max_pos{ positions_normalized[0] };
		for (const glm::vec3& pos : positions_normalized) {
			min_pos = glm::min(min_pos, pos);
			max_pos = glm::max(max_pos, pos);
		}
		glm::vec3 center{ (min_pos + max_pos) / 2.0f };

		// transform positions coords so that model is centered at origin
		for (glm::vec3& pos : positions_normalized)
		{
			pos -= center;
		}
	}
	
	// compute vertex normals if not present in file
	if (normals.empty() && load_nml_coord_flag)
	{
		normals.resize(positions_normalized.size(), glm::vec3{ 0.f,0.f,0.f });

		for (const Face& face : faces)
		{
			glm::vec3 v0{ positions_normalized[face.vertexIndices[0]] };
			glm::vec3 v1{ positions_normalized[face.vertexIndices[1]] };
			glm::vec3 v2{ positions_normalized[face.vertexIndices[2]] };

			glm::vec3 edge1{ v1 - v0 };
			glm::vec3 edge2{ v2 - v0 };

			// compute face normal of current triangle
			glm::vec3 face_normal
			{
				edge1.y * edge2.z - edge1.z * edge2.y,
				edge1.z * edge2.x - edge1.x * edge2.z,
				edge1.x * edge2.y - edge1.y * edge2.x
			};

			float length{ std::sqrtf(face_normal.x * face_normal.x +
				face_normal.y * face_normal.y +
				face_normal.z * face_normal.z) };

			if (length > 0)
			{
				face_normal.x /= length;
				face_normal.y /= length;
				face_normal.z /= length;
			}

			normals[face.vertexIndices[0]] += face_normal;
			normals[face.vertexIndices[1]] += face_normal;
			normals[face.vertexIndices[2]] += face_normal;
		}

		// normalize computed normals
		for (glm::vec3& normal : normals)
		{
			float length = std::sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
			if (length > 0)
			{
				normal.x /= length;
				normal.y /= length;
				normal.z /= length;
			}
		}
	}
	return true;
}