///////////////////////////////////////////////////////////////////////////////
// File Name:      polygon.hpp
//
// Description:    Header file for defining class Polygon.
//
// IMPORTANT NOTE: This file should NOT be modified.
///////////////////////////////////////////////////////////////////////////////

#ifndef POLYGON_HPP
#define POLYGON_HPP

#include "shape.hpp"
#include <iostream>
#include <vector>

/**
* @brief Polygon is-a Shape:
*/
class Polygon : public Shape {
private:
  std::vector<Point> vertices; // Polygon has vertices
  static int count;    // how many polygons created?

public:
/**
* @brief Constructor for an Polygon shape.
* The ctor parses space- and comma-separated line starting with character 'P' and
* sets up general Shape part plus specialized polygon information.
* This function must use the Shape(std::string&) ctor to set up the
* general Shape information consisting of id, border and fill colors.
* Then, this ctor will add unknown number of the shape's vertices. 
* Further, the ctor will increment the static data member
* count that keeps track of how many polygons created.
*
* @param line Line of comma-separated text from input file
*/
  Polygon(std::string& line);

// Each Polygon is unique. Therefore, they cannot be default constructed
// nor can there be replicas of a Polygon object.
  Polygon() = delete;
  Polygon(Polygon const&) = delete;
  Polygon& operator=(Polygon const&) = delete;

  /**
  * @brief Destructor. 
  * Make sure you decrement the static counter count during destruction.
  */
  virtual ~Polygon(); 

/**
* @brief Accessor for the shape's vertices.
*
* @return Vertices of the shape.
*/
  std::vector<Point> & getVertices();

/**
* @brief Get the total number of polygon shapes.
* Note that this is a static member function that should
* return the current value of static data member gcount.
*
* @return The number of polygon shapes.
*/
  static int getCount();

/**
* @brief Prints the details of Polygon Shape using
* Shape::print_details() to print Shape details.
* In addition to the fields printed by Shape::print_details(),
* this function should print type of shape (Polygon Shape),
* and vertices separated by space.
*
* @example Your function must follow this format to print details of
* a shape:
* Id = 12
* Border = black
* Fill = none
* Type = Polygon Shape
* Vertices = 72,89 35,74 10,22 100,111
*
* Note: It can be hard to keep track of which derived class functions
* are meant to override which base class functions. Fortunately, we
* can get compiler help to check. We can explicitly declare that a
* function is meant to override. Assuming that the derived class
* functions were meant to override, we can say so by adding override.
* Explicit use of override is particularly useful in large, complicated
* class hierarchies.
*/
  virtual void print_details() const override;

/**
* @brief Get the center of a shape. You can assume that the center
* of polygon is arithmetic mean of all vertices.
* Ex: when vertices are 72,89 35,74 10,22 100,111,
* the center is a point with x = (72+35+10+100)/4 
* and y = (89+74+22+111)/4. 
* @return Center of the polygon computed from the vertices.
*/
  virtual Point getCenter() const override;

/**
* @brief Get the area of a shape. You can assume that area i) is signed and ii) is 
*  calculated for a non-self-intersecting polygon.
*  Formula for computing the area for such polygons can be taken from
*  https://en.wikipedia.org/wiki/Polygon#Area
* @return The signed area of the shape which is a non-self-intersecting polygon.
*/
  virtual double getArea() const override;
};

#endif // end of #ifndef POLYGON_HPP
