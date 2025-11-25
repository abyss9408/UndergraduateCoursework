///////////////////////////////////////////////////////////////////////////////
// File Name:      ellipse.cpp
//
// Author:         Bryan Ang Wei Ze
// Email:          bryanweize.ang@digipen.edu
//
// Description:    Definitions of member functions of derived class Ellipse.
///////////////////////////////////////////////////////////////////////////////

#include "ellipse.hpp"
#include <iostream>
#include <sstream>
#include <cmath>

int Ellipse::count = 0;

Ellipse::Ellipse(std::string& line) : Shape{line}
{
    std::stringstream line_stream{line};
    char comma;
    line_stream >> center.x >> comma >> center.y;
    line_stream >> a >> b;
    ++count;
}

Ellipse::~Ellipse()
{
    --count;
}

int Ellipse::getA() const
{
    return a;
}

int Ellipse::getB() const
{
    return b;
}

int Ellipse::getCount()
{
    return count;
}

void Ellipse::print_details() const
{
    std::cout << "Id = "<< getId() << '\n';
    std::cout << "Border = "<< getBorder() << '\n';
    std::cout << "Fill = "<< getFill() << '\n';
    std::cout << "Type = Ellipse Shape\n";
    std::cout << "Center = " << getCenter().x << "," << getCenter().y << '\n';
    std::cout << "a-length = " << getA() << '\n';
    std::cout << "b-length = " << getB() << '\n';
}

Point Ellipse::getCenter() const
{
    return center;
}

double Ellipse::getArea() const
{
    return a * b * M_PI;
}