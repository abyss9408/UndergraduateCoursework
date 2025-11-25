///////////////////////////////////////////////////////////////////////////////
// File Name:      process.cpp
//
// Author:         Bryan Ang Wei Ze
// Email:          bryanweize.ang@digipen.edu
//
// Description:    Methods to perform some processing on containers of
//                 type std::vector<Shape*>.
///////////////////////////////////////////////////////////////////////////////

#include "process.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>      
#include <set>

static bool compare_area(const Shape* const & lhs, const Shape* const & rhs) {
  return lhs->getArea() < rhs->getArea();
}

void parse_file(std::ifstream &ifs, std::vector<Shape*>& vs,
                std::vector<Shape*>& ves, std::vector<Shape*>& vps) {
  std::string line;
  while (std::getline(ifs, line))
  {
    char c = line[0];
    line = line.substr(2);

    if (c == 'E')
    {
      Shape *p = new Ellipse(line);
      vs.push_back(p);

      ves.push_back(p);
    }
    else if (c == 'P')
    {
      Shape *p = new Polygon(line);
      vs.push_back(p);

      vps.push_back(p);
    }
    else
      break;
  }
}

void print_records(std::vector<Shape*> const& vs) {
  for (const Shape* const& s : vs)
  {
    s->print_details();
    std::cout << '\n';
  }
}

void print_stats(std::vector<Shape*> const& vs) {
  std::cout << "Number of shapes = "<< vs.size() << '\n';
  
  double total_area{};
  for (const Shape* const& s : vs)
  {
    total_area += s->getArea();
  }
  std::cout << "The mean of the areas = "<< total_area / vs.size() << '\n';

  std::cout << "The sorted list of shapes (id,center,area) in ascending order of areas:\n";

  std::vector<Shape*> copy_vs{vs};
  std::sort(copy_vs.begin(), copy_vs.end(), compare_area);
  for (const Shape* const& s : copy_vs)
  {
    std::cout << s->getId() << "," << s->getCenter().x << "," << s->getCenter().y << 
    ","<< s->getArea() << '\n';
  }
}

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
void cleanup(std::vector<Shape*>& vs, std::vector<Shape*>& ves, std::vector<Shape*>& vps) {
  // delete memory previously allocated by operator new
  for (Shape *ps : vs)
    delete ps;

  // avoid double deletion!!!
  vs.clear();
  ves.clear();
  vps.clear();
}
