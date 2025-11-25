/*!************************************************************************
    \file point.hpp

    \author Bryan Ang Wei Ze

    \par DP email: bryanweize.ang\@digipen.edu

    \par Course: CSD1171 High-Level Programming 2

    \par Lab #5

    \date 09-02-2024
    
    \brief
    This header file defines a class named Point, declares constructors,
    member function operator overloads and non-member function operator
    overloads for the class Point
**************************************************************************/

#ifndef POINT_HPP
#define POINT_HPP

#include <iostream> // istream, ostream

/*!************************************************************************
    \namespace hlp2
**************************************************************************/
namespace hlp2 {
	
class Point {
public:
	/*!***********************************************************************
        \brief
        Declaration of default constructor that initializes x and y with 0.0
    **************************************************************************/
	Point();
    
    /*!***********************************************************************
        \brief
        Declaration of constructor that takes in an array of doubles and
        initializes x and y with the first two elements respectively

        \param[in] arr
        Array of doubles
    **************************************************************************/
	Point(const double* arr);

    /*!***********************************************************************
        \brief
        Declaration of constructor that takes in two doubles and initializes
        x and y respectively.

        \param[in] a
        Value to initialize x with

        \param[in] b
        Value to initialize y with
    **************************************************************************/
	Point(double a, double b);
    
	/*!***********************************************************************
        \brief
        Declaration of const member function operator[] overload that takes in
        an index and returns the corresponding coordinate.

        \param[in] index
        Index to be read

        \return
        Corresponding coordinate.
    **************************************************************************/
	const double& operator[](int index) const;

    /*!***********************************************************************
        \brief
        Declaration of member function operator[] overload that takes in an
        index and returns the corresponding coordinate.

        \param[in] index
        Index to be read

        \return
        Corresponding coordinate.
    **************************************************************************/
	double& operator[](int index);

     /*!***********************************************************************
        \brief
        Declaration of member function operator+= overload that adds two points

        \param[in] rhs
        Right operand of this overload

        \return
        The result of adding two points
    **************************************************************************/
	Point& operator+=(const Point& rhs);

    /*!***********************************************************************
        \brief
        Declaration of member function operator+= overload that adds a value
        to both the x and y coordinates

        \param[in] rhs
        Right operand of this overload

        \return
        The result of adding a value to both the x and y coordinates
    **************************************************************************/
	Point& operator+=(const double rhs);

    /*!***********************************************************************
        \brief
        Declaration of member function prefix operator++ overload that prefix
        increments the x and y coordinates

        \return
        The result of prefix increments of x and y coordinates
    **************************************************************************/
	Point& operator++();

    /*!***********************************************************************
        \brief
        Declaration of member function postfix operator++ overload that postfix
        increments the x and y coordinates

        \param
        To create a different function signature from member function prefix
        operator++ overload

        \return
        The result of postfix increments of x and y coordinates
    **************************************************************************/
	Point operator++(int);

    /*!***********************************************************************
        \brief
        Declaration of member function prefix operator-- overload that prefix
        decrements the x and y coordinates

        \return
        The result of prefix decrements of x and y coordinates
    **************************************************************************/
	Point& operator--();

    /*!***********************************************************************
        \brief
        Declaration of member function postfix operator-- overload that postfix
        decrements the x and y coordinates
    
        \param
        To create a different function signature from member function prefix
        operator-- overload

        \return
        The result of postfix decrements of x and y coordinates
    **************************************************************************/
	Point operator--(int);

private:
	double x; // The x-coordinate of a Point
	double y; // The y-coordinate of a Point
};
  
/*!***********************************************************************
    \brief
    Declaration of non member function operator% overload that rotates a Point
    object about the origin

    \param[in] lhs
    Point to be rotated about the origin (left operand)
    
    \param[in] rhs
    Angle of rotation in degrees (right operand)

    \return
    The Point after rotation about the origin
**************************************************************************/
Point operator%(const Point& lhs, double rhs);

/*!***********************************************************************
    \brief
    Declaration of non member function operator/ overload that evaluates the
    distance between two Points

    \param[in] lhs
    First Point (left operand)
    
    \param[in] rhs
    Second Point (right operand)

    \return
    Distance between two Points
**************************************************************************/
double operator/(const Point& lhs, const Point& rhs);

/*!***********************************************************************
    \brief
    Declaration of non member function operator+ overload that adds the
    corresponding coordinates of two Points

    \param[in] lhs
    First Point (left operand)
    
    \param[in] rhs
    Second Point (right operand)

    \return
    The result of adding two Points
**************************************************************************/
Point operator+(const Point& lhs, const Point& rhs);

/*!***********************************************************************
    \brief
    Declaration of non member function binary operator+ overload that adds a
    value to the coordinates of a Point

    \param[in] lhs
    The Point (left operand)
    
    \param[in] rhs
    Value (right operand)

    \return
    The result of adding a value to the coordinates of the Point
**************************************************************************/
Point operator+(const Point& lhs, double rhs);

/*!***********************************************************************
    \brief
    Declaration of non member function binary operator+ overload that adds a
    value to the coordinates of a Point

    \param[in] lhs
    Value (left operand)
    
    \param[in] rhs
    The Point (right operand)

    \return
    The result of adding a value to the coordinates of the Point
**************************************************************************/
Point operator+(double rhs, const Point& lhs);

/*!***********************************************************************
    \brief
    Declaration of non member function binary operator- overload that subtracts
    the corresponding coordinates of two Points

    \param[in] lhs
    First Point (left operand)
    
    \param[in] rhs
    Second Point (right operand)

    \return
    The result of subtracting two Points
**************************************************************************/
Point operator-(const Point& lhs, const Point& rhs);

/*!***********************************************************************
    \brief
    Declaration of non member function binary operator- overload that subtracts
    a value from the coordinates of a Point

    \param[in] lhs
    The Point (left operand)
    
    \param[in] rhs
    Value (right operand)

    \return
    The result of subtracting a value from the coordinates of the Point
**************************************************************************/
Point operator-(const Point& lhs, double rhs);

/*!***********************************************************************
    \brief
    Declaration of non member function binary operator- overload that subtracts
    the coordinates of a Point from a value

    \param[in] lhs
    Value (left operand)
    
    \param[in] rhs
    The Point (right operand)

    \return
    The result of subtracting the coordinates of the Point from a value
**************************************************************************/
Point operator-(double rhs, const Point& lhs);

/*!***********************************************************************
    \brief
    Declaration of non member function unary operator- overload that negates
    the coordinates of a Point

    \param[in] rhs
    The Point (right operand)

    \return
    The result of negating the coordinates of the Point
**************************************************************************/
Point operator-(const Point& rhs);

/*!***********************************************************************
    \brief
    Declaration of non member function operator^ overload that returns
    the Point midway between two Points

    \param[in] lhs
    First Point (left operand)

    \param[in] rhs
    Second Point (right operand)

    \return
    The Point midway between two Points
**************************************************************************/
Point operator^(const Point& lhs, const Point& rhs);

/*!***********************************************************************
    \brief
    Declaration of non member function operator* overload that returns the
    dot product of two Points

    \param[in] lhs
    First Point (left operand)

    \param[in] rhs
    Second Point (right operand)

    \return
    The dot product of two points
**************************************************************************/
double operator*(const Point& lhs, const Point& rhs);

/*!***********************************************************************
    \brief
    Declaration of non member function operator* overload that returns the
    scaling of a Point

    \param[in] lhs
    Scalar (left operand)

    \param[in] rhs
    The Point (right operand)

    \return
    The result of scalar multiplication of the Point
**************************************************************************/
Point operator*(double lhs, const Point& rhs);

/*!***********************************************************************
    \brief
    Declaration of non member function operator* overload that returns the
    scaling of a Point

    \param[in] lhs
    The Point (left operand)

    \param[in] rhs
    Scalar (right operand)

    \return
    The result of scalar multiplication of the Point
**************************************************************************/
Point operator*(const Point& lhs, double rhs);

/*!***********************************************************************
    \brief
    Declaration of non member function operator<< overload that writes to output
    stream coordinates of Point

    \param[in] os
    The output stream (left operand)

    \param[in] rhs
    The Point (right operand)

    \return
    The output stream
**************************************************************************/
std::ostream& operator<<(std::ostream& os, const Point& rhs);

/*!***********************************************************************
    \brief
    Declaration of non member function operator>> overload that reads two
    doubles as Point from input stream

    \param[in] os
    The input stream (left operand)

    \param[in] rhs
    The Point (right operand)

    \return
    The input stream
**************************************************************************/
std::istream& operator>>(std::istream& is, Point& rhs);
} // end namespace hlp2

#endif // end POINT_HPP
////////////////////////////////////////////////////////////////////////////////
