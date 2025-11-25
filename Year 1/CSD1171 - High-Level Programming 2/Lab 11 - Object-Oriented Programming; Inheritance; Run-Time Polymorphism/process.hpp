///////////////////////////////////////////////////////////////////////////////
// File Name:      process.hpp
//
// Description:    Methods to perform some processing on shape objects.
//
// IMPORTANT NOTE: This file should NOT be modified.
///////////////////////////////////////////////////////////////////////////////

#ifndef PROCESS_HPP
#define PROCESS_HPP

#include "shape.hpp"
#include "ellipse.hpp"
#include "polygon.hpp"
#include <fstream>
#include <vector>

/**
* @brief Parse space- and comma-separated file containing shapes' records - one line
* per shape. Each line of text contains a space- and comma-separated list of information
* common to all shapes.
* The function will read a line, determine whether the line pertains to a
* Ellipse or Polygon shape.
* If first non-whitespace character in file is E, then the line contains
* information pertaining to an ellipse; otherwise, if the first non-white
* space character is P, then the line contains information pertaining to a polygon
* shape.
* Depending on the type of Shape, the appropriate ctor is used to initialize
* either a Polygon or Ellipse object on the free store.
* Since Polygon is-a Shape and Ellipse is-a Shape, the value returned
* by operator new is assigned to a variable of type Shape*.
* This pointer is added to a vector container that will be the repository of
* ALL shapes.
* In addition, pointers to free store objects of type Polygon are added to a
* vector container that will be the repository of ONLY polygon shapes. Likewise,
* pointers to free store objects of type Ellipse are added to a vector
* container that will be the repository of ONLY ellipse shapes. This is
* a common idiom: initialize an object on the free store only once with multiple
* containers containing pointers to that free store object.
*
* @param ifs Input file stream with shapes' data.
* @param vs  Reference to vector<Shape*> that will be filled with
*                   pointers to every shape in file.
* @param ves Reference to vector<Shape*> that will be filled with
*                   pointers to ellipse shapes in file.
* @param vps Reference to vector<Shape*> that will be filled with
*                   pointers to polygon shapes in file.
*/
void parse_file(std::ifstream& ifs,
                std::vector<Shape*>& vs,
                std::vector<Shape*>& ves,
                std::vector<Shape*>& vps);

/**
* @brief Use dynamic-binding (run-time polygmorphism) to print details of
*        shapes represented in std::vector<Shape*> container.
*
* @param vs Read-only reference to std::vector<Shape*>.
*/
void print_records(std::vector<Shape*> const& vs);

/**
* @brief Computes and prints following statistics: shape count; mean of
*        total areas; and sorted list of shapes in ASCENDING order using 
*        a shape's area as sorting criterion with each line  
*        consisting of the following information:
*        shape id, center, area.
*
* @param vs Read-only reference to vector<Shape*> for which statistics
*                 are to be computed.
*/
void print_stats(std::vector<Shape*> const& vs);

/**
* @brief Return memory allocated by operator new ONLY using pointers contained
* in container specified by first parameter. Why? Because containers specified
* in the next two containers reference the same memory locations as elements of
* first container, double deletion will cause undefined run-time behavior.
* Next, clear contents of all 3 containers.
*
* @param vs Reference to vector<Shape*> containing pointers to all shapes.
* @param ves Reference to vector<Shape*> containing pointers to ellipses.
* @param vps Reference to vector<Shape*> containing pointers to polygons.
*/
void cleanup(std::vector<Shape*>& vs, 
             std::vector<Shape*>& ves,
             std::vector<Shape*>& vps);

#endif // end #ifndef PROCESS_HPP
