///////////////////////////////////////////////////////////////////////////////
// File Name:      ellipse.hpp
//
// Description:    The header file for defining class Ellipse.
//
// IMPORTANT NOTE: This file should NOT be modified.
///////////////////////////////////////////////////////////////////////////////

#ifndef ELLIPSE_HPP
#define ELLIPSE_HPP

#include "shape.hpp"
#include <string>

/**
* @brief Ellipse is-a Shape:
*/
class Ellipse : public Shape {
private:
  Point center; 
  int a;   // a-length, length of the semi-major axis
  int b;   // b-length, length of the semi-minor axis
  static int count; // how many ellipses created?

public:
/**
* @brief Constructor for an Ellipse shape.
* The ctor parses space- and comma-separated line starting with character 'E' and
* sets up general Shape part plus specialized ellipse information.
* This function must use the Shape(std::string&) ctor to set up the
* general Shape information consisting of id, border color, fill color.
* Then, this ctor will add the center point position and length of a and b axes. 
* Further, the ctor will increment the static data member count that keeps track
* of how many ellipses are created.
*
* @param line Line of comma-separated text from input file
*/
  Ellipse(std::string& line);

// Each Ellipse is unique. Therefore, they cannot be default constructed
// nor can there be replicas of an Ellipse object.
  Ellipse() = delete;
  Ellipse(Ellipse const&) = delete;
  Ellipse& operator=(Ellipse const&) = delete;

  /**
  * @brief Destructor. 
  * Make sure you decrement the static counter count during destruction.
  */  
  virtual ~Ellipse();

/**
* @brief Accessor for a shape's a-length.
*
* @return The a-length of the shape.
*/
  int getA() const;

/**
* @brief Accessor for a shape's b-length.
*
* @return The b-length of the shape.
*/
  int getB() const;

/**
* @brief Get the total number of ellipse shapes.
* Note that this is a static member function that should
* return the current value of static data member count.
*
* @return The number of ellipse shapes.
*/
  static int getCount();

/**
* @brief Prints the details of Ellipse Shape using
* Shape::print_details() to print Shape details.
* In addition to the fields printed by Shape::print_details(), this function
* should print type of shape (Ellipse Shape), center point and
* a- and b-lengths
*
* @example Your function must follow this format to print details of
* a shape:
* Id = 12
* Border = black
* Fill = none
* Type = Ellipse Shape
* Center = 72,89
* a-length = 5
* b-length = 10
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
* @brief Accessor for a shape's center. 
* @return Center of the ellipse. The point in which the shape is located.
*/
  virtual Point getCenter() const override;

/**
* @brief Get the area of a shape.
*  Formula for computing the area for ellipse is:
*                     a * b * PI
* @return The area of the shape.
*/
  virtual double getArea() const override;
};

#endif // end #ifndef ELLIPSE_HPP
