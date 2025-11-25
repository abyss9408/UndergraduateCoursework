/*!*****************************************************************************
\file unittests_functions.h
\author Vadim Surov (vsurov\@digipen.edu)
\co-author YOUR NAME (DIGIPEN ID)
\par Course: CSD2151/CSD2150/CS250
\par Assignment: 4.1 (BlinnPhong)
\date 02/06/2024 (MM/DD/YYYY)
\brief This file has declarations of functions for unit tests.

This code is intended to be completed and submitted by a student,
so any changes in this file will be used in the evaluation on the VPL server.
You should not change functions' name and parameter types in provided code.
*******************************************************************************/
#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "unittests_data.h"

/*
    Read specs for Assignment 4.1
*/
vec3 BlinnPhong(vec3 position, vec3 normal, Light light, Material material, mat4 view);

#endif