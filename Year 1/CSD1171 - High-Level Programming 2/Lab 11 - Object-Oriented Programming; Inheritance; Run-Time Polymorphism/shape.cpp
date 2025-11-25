///////////////////////////////////////////////////////////////////////////////
// File Name:      shape.cpp
//
// Author:         Bryan Ang Wei Ze
// Email:          bryanweize.ang@digipen.edu
//
// Description:    Definitions of member functions of base class Shape.
///////////////////////////////////////////////////////////////////////////////

#include "shape.hpp"
#include <iostream>

int Shape::count = 0;

Shape::Shape(std::string& line)
{
    size_t border_end{}, fill_end{};

    // increment static member count
    ++count;
    id = count;

    // extract border color
    border_end = line.find_first_of(' ');
    border = line.substr(0, border_end);

    // remove border color portion from line
    line = line.substr(border_end+1);

    // extract fill color
    fill_end = line.find_first_of(' ');
    fill = line.substr(0, fill_end);

    // remove fill color portion from line
    line = line.substr(fill_end+1);
}

Shape::~Shape()
{
    --count;
}

int Shape::getId() const
{
    return id;
}

std::string Shape::getBorder() const
{
    return border;
}

std::string Shape::getFill() const
{
    return fill;
}

int Shape::getCount()
{
    return count;
}

void Shape::print_details() const
{
    std::cout << "Id = "<< id << '\n';
    std::cout << "Border = "<< border << '\n';
    std::cout << "Fill = "<< fill << '\n';
}