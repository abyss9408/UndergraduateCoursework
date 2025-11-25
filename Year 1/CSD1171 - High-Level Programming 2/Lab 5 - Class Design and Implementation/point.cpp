/*!************************************************************************
    \file point.cpp

    \author Bryan Ang Wei Ze

    \par DP email: bryanweize.ang\@digipen.edu

    \par Course: CSD1171 High-Level Programming 2

    \par Lab #5

    \date 09-02-2024
    
    \brief
    This source file implements constructors, member function operator overloads
    and non-member function operator overloads for the class Point
**************************************************************************/

#include "point.hpp"  // Point interface
#include <cmath>      // sin, cos, sqrt

namespace {
    const double PI {3.14159265358979323846};
}

/*!************************************************************************
    \namespace hlp2
**************************************************************************/
namespace hlp2 {

/*!***********************************************************************
    \brief
    Implementation of default constructor that initializes x and y with 0.0
**************************************************************************/
Point::Point() : x{0.0}, y{0.0}
{

}

/*!***********************************************************************
    \brief
    Implementation of constructor that takes in an array of doubles and
    initializes x and y with the first two elements respectively

    \param[in] arr
    Array of doubles
**************************************************************************/
Point::Point(const double* arr) : x{*arr}, y{*(arr+1)}
{

}

/*!***********************************************************************
    \brief
    Implementation of constructor that takes in two doubles and initializes
    x and y respectively.

    \param[in] a
    Value to initialize x with

    \param[in] b
    Value to initialize y with
**************************************************************************/
Point::Point(double a, double b) : x{a}, y{b}
{

}

/*!***********************************************************************
    \brief
    Implementation of const member function operator[] overload that takes in
    an index and returns the corresponding coordinate.

    \param[in] index
    Index to be read

    \return
    Corresponding coordinate.
**************************************************************************/
const double& Point::operator[](int index) const
{   
    return index ? y : x;
}

/*!***********************************************************************
    \brief
    Implementation of member function operator[] overload that takes in an
    index and returns the corresponding coordinate.

    \param[in] index
    Index to be read

    \return
    Corresponding coordinate.
**************************************************************************/
double& Point::operator[](int index)
{   
    return index ? y : x;
}

/*!***********************************************************************
    \brief
    Implementation of member function operator+= overload that adds two points

    \param[in] rhs
    Right operand of this overload

    \return
    The result of adding two points
**************************************************************************/
Point& Point::operator+=(const Point& rhs)
{
    x += rhs.x;
    y += rhs.y;
    return *this;
}

/*!***********************************************************************
    \brief
    Implementation of member function operator+= overload that adds a value
    to both the x and y coordinates

    \param[in] rhs
    Right operand of this overload

    \return
    The result of adding a value to both the x and y coordinates
**************************************************************************/
Point& Point::operator+=(const double rhs)
{
    x += rhs;
    y += rhs;
    return *this;
}

/*!***********************************************************************
    \brief
    Implementation of member function prefix operator++ overload that prefix
    increments the x and y coordinates

    \return
    The result of prefix increments of x and y coordinates
**************************************************************************/
Point& Point::operator++()
{
    ++x;
    ++y;
    return *this;
}

/*!***********************************************************************
    \brief
    Implementation of member function postfix operator++ overload that postfix
    increments the x and y coordinates

    \param
    To create a different function signature from member function prefix
    operator++ overload

    \return
    The result of postfix increments of x and y coordinates
**************************************************************************/
Point Point::operator++(int)
{
    Point tmp{*this};
    ++x;
    ++y;
    return tmp;
}

/*!***********************************************************************
    \brief
    Implementation of member function prefix operator-- overload that prefix
    decrements the x and y coordinates

    \return
    The result of prefix decrements of x and y coordinates
**************************************************************************/
Point& Point::operator--()
{
    --x;
    --y;
    return *this;
}

/*!***********************************************************************
    \brief
    Implementation of member function postfix operator-- overload that postfix
    decrements the x and y coordinates
    
    \param
    To create a different function signature from member function prefix
    operator-- overload

    \return
    The result of postfix decrements of x and y coordinates
**************************************************************************/
Point Point::operator--(int)
{
    Point tmp{*this};
    --x;
    --y;
    return tmp;
}

/*!***********************************************************************
    \brief
    Implementation of non member function operator% overload that rotates a Point
    object about the origin

    \param[in] lhs
    Point to be rotated about the origin (left operand)
    
    \param[in] rhs
    Angle of rotation in degrees (right operand)

    \return
    The Point after rotation about the origin
**************************************************************************/
Point operator%(const Point& lhs, double rhs)
{
    Point tmp;
    double radians{rhs*PI/180};
    tmp[0] = lhs[0] * cos(radians) - lhs[1] * sin(radians);
    tmp[1] = lhs[1] * cos(radians) + lhs[0] * sin(radians);
    return tmp;
}

/*!***********************************************************************
    \brief
    Implementation of non member function operator/ overload that evaluates the
    distance between two Points

    \param[in] lhs
    First Point (left operand)
    
    \param[in] rhs
    Second Point (right operand)

    \return
    Distance between two Points
**************************************************************************/
double operator/(const Point& lhs, const Point& rhs)
{
    return sqrt((lhs[0] - rhs[0])*(lhs[0] - rhs[0]) + (lhs[1] - rhs[1])*(lhs[1] - rhs[1]));
}

/*!***********************************************************************
    \brief
    Implementation of non member function operator+ overload that adds the
    corresponding coordinates of two Points

    \param[in] lhs
    First Point (left operand)
    
    \param[in] rhs
    Second Point (right operand)

    \return
    The result of adding two Points
**************************************************************************/
Point operator+(const Point& lhs, const Point& rhs)
{
    Point tmp;
    tmp[0] = lhs[0] + rhs[0];
    tmp[1] = lhs[1] + rhs[1];
    return tmp;
}

/*!***********************************************************************
    \brief
    Implementation of non member function binary operator+ overload that adds a
    value to the coordinates of a Point

    \param[in] lhs
    The Point (left operand)
    
    \param[in] rhs
    Value (right operand)

    \return
    The result of adding a value to the coordinates of the Point
**************************************************************************/
Point operator+(const Point& lhs, double rhs)
{
    Point tmp;
    tmp[0] = lhs[0] + rhs;
    tmp[1] = lhs[1] + rhs;
    return tmp;
}

/*!***********************************************************************
    \brief
    Implementation of non member function binary operator+ overload that adds a
    value to the coordinates of a Point

    \param[in] lhs
    Value (left operand)
    
    \param[in] rhs
    The Point (right operand)

    \return
    The result of adding a value to the coordinates of the Point
**************************************************************************/
Point operator+(double lhs, const Point& rhs)
{
    return operator+(rhs, lhs);
}

/*!***********************************************************************
    \brief
    Implementation of non member function binary operator- overload that subtracts
    the corresponding coordinates of two Points

    \param[in] lhs
    First Point (left operand)
    
    \param[in] rhs
    Second Point (right operand)

    \return
    The result of subtracting two Points
**************************************************************************/
Point operator-(const Point& lhs, const Point& rhs)
{
    Point tmp;
    tmp[0] = lhs[0] - rhs[0];
    tmp[1] = lhs[1] - rhs[1];
    return tmp;
}

/*!***********************************************************************
    \brief
    Implementation of non member function binary operator- overload that subtracts
    a value from the coordinates of a Point

    \param[in] lhs
    The Point (left operand)
    
    \param[in] rhs
    Value (right operand)

    \return
    The result of subtracting a value from the coordinates of the Point
**************************************************************************/
Point operator-(const Point& lhs, double rhs)
{
    Point tmp;
    tmp[0] = lhs[0] - rhs;
    tmp[1] = lhs[1] - rhs;
    return tmp;
}

/*!***********************************************************************
    \brief
    Implementation of non member function binary operator- overload that subtracts
    the coordinates of a Point from a value

    \param[in] lhs
    Value (left operand)
    
    \param[in] rhs
    The Point (right operand)

    \return
    The result of subtracting the coordinates of the Point from a value
**************************************************************************/
Point operator-(double lhs, const Point& rhs)
{
    Point tmp;
    tmp[0] = lhs - rhs[0];
    tmp[1] = lhs - rhs[1];
    return tmp;
}

/*!***********************************************************************
    \brief
    Implementation of non member function unary operator- overload that negates
    the coordinates of a Point

    \param[in] rhs
    The Point (right operand)

    \return
    The result of negating the coordinates of the Point
**************************************************************************/
Point operator-(const Point& rhs)
{
    Point tmp;
    tmp[0] = -rhs[0];
    tmp[1] = -rhs[1];
    return tmp;
}

/*!***********************************************************************
    \brief
    Implementation of non member function operator^ overload that returns
    the Point midway between two Points

    \param[in] lhs
    First Point (left operand)

    \param[in] rhs
    Second Point (right operand)

    \return
    The Point midway between two Points
**************************************************************************/
Point operator^(const Point& lhs, const Point& rhs)
{
    Point tmp;
    tmp[0] = (lhs[0] + rhs[0])/2;
    tmp[1] = (lhs[1] + rhs[1])/2;
    return tmp;
}

/*!***********************************************************************
    \brief
    Implementation of non member function operator* overload that returns the
    dot product of two Points

    \param[in] lhs
    First Point (left operand)

    \param[in] rhs
    Second Point (right operand)

    \return
    The dot product of two points
**************************************************************************/
double operator*(const Point& lhs, const Point& rhs)
{
    return lhs[0] * rhs[0] + lhs[1] * rhs[1];
}

/*!***********************************************************************
    \brief
    Implementation of non member function operator* overload that returns the
    scaling of a Point

    \param[in] lhs
    Scalar (left operand)

    \param[in] rhs
    The Point (right operand)

    \return
    The result of scalar multiplication of the Point
**************************************************************************/
Point operator*(double lhs, const Point& rhs)
{
    Point tmp{rhs};
    tmp[0] = lhs * rhs[0];
    tmp[1] = lhs * rhs[1];
    return tmp;
}

/*!***********************************************************************
    \brief
    Implementation of non member function operator* overload that returns the
    scaling of a Point

    \param[in] lhs
    The Point (left operand)

    \param[in] rhs
    Scalar (right operand)

    \return
    The result of scalar multiplication of the Point
**************************************************************************/
Point operator*(const Point& lhs, double rhs)
{
    Point tmp{lhs};
    tmp[0] = lhs[0] * rhs;
    tmp[1] = lhs[1] * rhs;
    return tmp;
}

/*!***********************************************************************
    \brief
    Implementation of non member function operator<< overload that writes to output
    stream coordinates of Point

    \param[in] os
    The output stream (left operand)

    \param[in] rhs
    The Point (right operand)

    \return
    The output stream
**************************************************************************/
std::ostream& operator<<(std::ostream& os, const Point& rhs)
{
    os << "(" << rhs[0] << ", "<<rhs[1]<<")";
    return os;
}

/*!***********************************************************************
    \brief
    Implementation of non member function operator>> overload that reads two
    doubles as Point from input stream

    \param[in] os
    The input stream (left operand)

    \param[in] rhs
    The Point (right operand)

    \return
    The input stream
**************************************************************************/
std::istream& operator>>(std::istream& is, Point& rhs)
{
    is >> rhs[0];
    is >> rhs[1];
    return is;
}
} // end hlp2 namespace
