///////////////////////////////////////////////////////////////////////////////
// File Name:      shape.hpp
//
// Description:    The header file for class Shape.
//
// IMPORTANT NOTE: You will not submit this file - therefore you should NOT 
//                 modify this file.
///////////////////////////////////////////////////////////////////////////////

#ifndef SHAPE_HPP
#define SHAPE_HPP

#include <string>

/**
* @brief A struct for a point in two-dimentional descrete space.
*/
struct Point {
  int x;
  int y;
};

/**
* @brief An abstract base class for a Shape.
*/
class Shape {
private:
  int id;                  // a shape's unique id.
  std::string border;      // a shape's border color
  std::string fill;        // a shape's fill color
  static int count;        // how many shapes created?

public:
/**
* @brief A constructor for a Shape that parses shape information from one
* line of space-separated file.
* This ctor will parse the line for information common to all shapes:
* border and fill color.
* Ex: black none
* If the ctor's parameter consists of this sequence of characters:
*    black white 0,0 10 20
* the ctor must extract this common information:  
* black (border color), white (fill color) and assign this 
* information to the appropriate data members.
* Extracted information MUST be deleted from the sequence in order to 
* correctly process the following information (0,0 10 20) by the derived class.
*
* In addition, the function must increment static data member Shape::count.
* Note that the input file doesn't specify a shape id. Instead, this ctor will
* generate a monotonically increasing ID for a shape based on their order in
* the input file. The IDs are non-zero and start with 1. That is, the first
* shape encountered will be assigned an ID of 1, the second shape encountered
* will be assigned an ID of 2, and so on. 
* (Tip: easiest way to do it is to use static count).
*
* @param line Line of comma-separated text from input file
*/
  Shape(std::string& line);

// Each Shape is unique. Therefore, they cannot be default constructed
// nor can there be replicas of a Shape.
  Shape() = delete;
  Shape(Shape const&) = delete;
  Shape& operator=(Shape const&) = delete;

  /**
  * @brief Destructor. 
  * Make sure you decrement the static counter count during destruction.
  */
  virtual ~Shape();

/**
* @brief Accessor for shape's id.
*
* @return The id of the shape.
*/
  int getId() const;

/**
* @brief Accessor for shape's border color.
*
* @return The border color of the shape.
*/
  std::string getBorder() const;

/**
* @brief Accessor for shape's fill color.
*
* @return The fill color of the shape.
*/
  std::string getFill() const;

/**
* @brief Get the count of shapes.
* Note that this is a static member function that should
* return the current value of static data member count.
*
* @return Shape count.
*/
  static int getCount();

/**
* @brief Prints the details of Shape.
* This method prints the common information about a shape.
* 
* @example Your function must follow this format to print details of
* a shape:
* Id = 12
* Border = black
* Fill = none
*/
  virtual void print_details() const;

/**
* @brief Get the center of a shape. You can assume that the center
* of polygon is arithmetic mean of all vertices.
* Shape class will not define this function!!! Instead, since this is a
* pure virtual function, the derived class will implement this function. 
* @return Center of the shape computed from the shape parameters.
*/
  virtual Point getCenter() const = 0;

/**
* @brief Get the area of a shape. 
* Shape class will not define this function!!! Instead, since this is a
* pure virtual function, the derived class will implement this function. 
* @return Area of shape computed from the shape parameters.
*/
  virtual double getArea() const = 0;

};

#endif // end #ifndef SHAPE_HPP
