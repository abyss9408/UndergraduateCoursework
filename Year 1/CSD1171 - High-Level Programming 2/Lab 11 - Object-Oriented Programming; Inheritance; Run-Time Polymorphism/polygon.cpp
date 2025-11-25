///////////////////////////////////////////////////////////////////////////////
// File Name:      polygon.cpp
//
// Author:         Bryan Ang Wei Ze
// Email:          bryanweize.ang@digipen.edu
//
// Description:    Definitions of member functions of derived class Polygon.
///////////////////////////////////////////////////////////////////////////////

#include "polygon.hpp"
#include <iostream>
#include <sstream>

int Polygon::count = 0;

Polygon::Polygon(std::string& line) : Shape{line}
{
    std::stringstream line_stream{line};
    char comma;
    int x{}, y{};
    while (line_stream.good())
    {
        line_stream >> x >> comma >> y;
        vertices.push_back({x, y});
    }
    ++count;
}

Polygon::~Polygon()
{
    --count;
}

std::vector<Point>& Polygon::getVertices()
{
    return vertices;
}

int Polygon::getCount()
{
    return count;
}

void Polygon::print_details() const
{
    std::cout << "Id = "<< getId() << '\n';
    std::cout << "Border = "<< getBorder() << '\n';
    std::cout << "Fill = "<< getFill() << '\n';
    std::cout << "Type = Polygon Shape\n";
    std::cout << "Vertices = ";

    for (const Point& p : vertices)
    {
        std::cout << p.x << "," << p.y << " ";
    }
    
    std::cout << '\n';
}

Point Polygon::getCenter() const
{
    int sum_x{}, sum_y{};
    size_t sides{};
    Point result{};

    sides = vertices.size();

    for (const Point& p : vertices)
    {
        sum_x += p.x;
        sum_y += p.y;
    }
    
    result.x = sum_x / sides;
    result.y = sum_y / sides;

    return result;
}

double Polygon::getArea() const
{
    double area{};
    size_t sides{};
    sides = vertices.size();

    // shoelace formula implementation
    size_t i;
    for (i = 0; i < sides - 1; ++i)
    {
        area += vertices[i].x * vertices[i + 1].y - vertices[i].y * vertices[i + 1].x;
    }
    area += vertices[i].x * vertices[0].y - vertices[i].y * vertices[0].x;

    return area / 2.0;
}