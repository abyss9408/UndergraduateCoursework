/*!
@file       objparser.cpp
@author     bryanweize.ang@digipen.edu
@date       10/07/2024

This file declares a function that processes .obj files

*//*__________________________________________________________________________*/

#ifndef OBJPARSER_H
#define OBJPARSER_H

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <glm/glm.hpp>

namespace MyObjParser
{
  /*  _________________________________________________________________________*/
  /*! parse_obj_mesh
  This function parses an OBJ geometry file and stores the contents of the file
  as array of vertex, array of normal , and an array of texture coordinate data.
  These three arrays will have the same size.
  Triangles are defined as an array of indices into array of position
  coordinates.

  @param std::string filename
  The name of the file containing the OBJ geometry information.

  @param std::vector<glm::vec3>& positions_normalized
  Fill user-supplied container with normalized vertex position attributes.
  The function WILL assume that the container is empty!!!

  @param std::vector<glm::vec3>& normals
  Fill user-supplied container with vertex normal attributes.
  The function WILL assume that the container is empty!!!
  If filled, this container will be the same size as "positions".

  @param std::vector<glm::vec2>& texcoords
  Fill user-supplied container with vertex texture coordinate attributes.
  The function WILL assume that the container is empty!!!
  If filled, this container will be the same size as "positions".

  @param std::vector<unsigned short>& triangles
  Triangle vertices are specified as indices into containers "positions",
  "normals", and "texcoords". Triangles will always have counter-clockwise
  orientation. This means that when looking at a face from the outside of
  the box, the triangles are counter-clockwise oriented.

  @param bool load_tex_coord_flag
  If parameter is true, then texture coordinates (if present in file) will
  be parsed. Otherwise, texture coordinate (even if present in file) will
  not be read.

  @param bool load_nml_coord_flag
  If parameter is true, then per-vertex normal coordinates
  will be parsed if they are present in file, otherwise, the per-vertex
  normals are computed.
  If the parameter is false, normal coordinate will neither be read from
  file (if present) nor explicitly computed.

  @param bool model_centered_flag
  In some cases, the modeler might have generated the model such that the
  center (of gravity) of the model is not centered at the origin.
  If the parameter is true, then the function will compute an axis-aligned
  bounding box and translate the position coordinates so that the box's center
  is at the origin.
  If the parameter is false, the position coordinates are left untouched.

  @return bool
  true if successful, otherwise false.
  The function will return false if the file is not present.
  */
	bool parse_obj_mesh(std::string filename, std::vector<glm::vec3>& positions_normalized,
		std::vector<glm::vec3>& normals, std::vector<glm::vec2>& tex_coords,
		std::vector<unsigned short>& triangles, bool load_nml_coord_flag, bool load_tex_coord_flag,
		bool model_centered_flag = true);

	struct Face
	{
		unsigned short vertexIndices[3];
		unsigned short textureIndices[3];
		unsigned short normalIndices[3];
	};
}

#endif
