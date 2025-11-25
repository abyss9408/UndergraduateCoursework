///////////////////////////////////////////////////////////////////////////////
// File Name:      vpl-driver.cpp
//
// Description:    The main program for Lab on OOP, inheritance, and
//                 run-time polymorphism.
//
// IMPORTANT NOTE: This file should NOT be modified.
///////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "shape.hpp"
#include "ellipse.hpp"
#include "polygon.hpp"
#include "process.hpp"

// static function declarations to print shape details ...
static void print_all(std::vector<Shape*> const&);
static void print_polygon(std::vector<Shape*> const&);
static void print_ellipse(std::vector<Shape*> const&);

/**
 * @brief The program execution begins here.
 * @param argc The number of command-line arguments.
 * @param argv The command line arguments: there should be one ore two arguments.
 * The first argument is the program name (that we'll ignore).
 * The second determines the type of output:
 * 1: prints information about ellipses including statistics
 * 2: prints information about polygon shapes including statistics.
 * 3: prints information about ALL shapes including statistics
 * 4: prints all the information specified by arguments 1, 2, and 3.
 * If second argument is empty, then input from cin is used to set it.
 * @return 0 for normal program termination, -1 for abnormal termination.
 */
int main(int argc, char *argv[]) {

  int output_type {1};

  if (argc == 1) 
    std::cin >> output_type;
  else if (argc == 2) 
    output_type = std::stoi(std::string(argv[1]));

  output_type = (output_type < 1) ? 1 : (output_type > 4) ? 4 : output_type;

  // is file specified on command-line valid?
  std::ifstream ifs{"shapes.txt"};
  if (!ifs) {
    std::cerr << "Unable to open input file shapes.txt!!!\n";
    return 0;
  }

  // parse file and populate containers with pointers to derived class objects
  std::vector<Shape*> vs;  // container for (pointers to) all shapes
  std::vector<Shape*> ves; // container for (pointers to) ellipses
  std::vector<Shape*> vps; // container for (pointers to) polygons
  parse_file(ifs, vs, ves, vps);

  std::cout << "Total number of shapes in file         = "
            << Shape::getCount() << "\n";
  std::cout << "Total number of ellipse shapes in file = "
            << Ellipse::getCount() << "\n";
  std::cout << "Total number of polygon shapes in file = "
            << Polygon::getCount() << "\n\n";
  
  switch (output_type) {
    case 1: print_ellipse(ves); break;
    case 2: print_polygon(vps); break;
    case 3: print_all(vs); break;
    case 4: print_all(vs); print_ellipse(ves); print_polygon(vps); break;
  }

  cleanup(vs, ves, vps);
}

// print all information about shapes in course
static void print_all(std::vector<Shape*> const& vs) {
  std::cout << "SHAPE INFORMATION" << "\n";
  std::cout << "---------------------------------" << "\n";
  print_records(vs);

  std::cout << "SHAPE STATISTICS" << "\n";
  std::cout << "--------------------------------" << "\n";
  print_stats(vs);
}

static void print_ellipse(std::vector<Shape*> const& ves) {
  std::cout << "ELLIPSE SHAPE INFORMATION" << "\n";
  std::cout << "---------------------------------" << "\n";
  print_records(ves);

  std::cout << "ELLIPSE SHAPE STATISTICS" << "\n";
  std::cout << "--------------------------------" << "\n";
  print_stats(ves);
}

static void print_polygon(std::vector<Shape*> const& vps) {
  std::cout << "POLYGON SHAPE INFORMATION" << "\n";
  std::cout << "----------------------------" << "\n";
  print_records(vps);

  std::cout << "POLYGON SHAPE STATISTICS" << "\n";
  std::cout << "--------------------------------" << "\n";
  print_stats(vps);
}
