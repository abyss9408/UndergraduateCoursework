/*!************************************************************************
  \file grad.cpp

  \author Bryan Ang Wei Ze

  \par DP email: bryanweize.ang\@digipen.edu

  \par Course: CSD2126 Modern C++ Design Patterns

  \par Programming Assignment #4

  \date 26-10-2024
  
  \brief
  This source file contains definition of class Graduate member functions
**************************************************************************/
#include "grad.hpp"
#include <iostream>
#include <sstream>

int Graduate::gcount = 0;

Graduate::Graduate(std::string& line) : Student(line)
{
    size_t research_end{};

    // extract research area
    research_end = line.find_first_of(',');
    gresearch = line.substr(0, research_end);

    // remove research area portion from line
    line = line.substr(research_end + 2);

    // extract advisor
    gadvisor = line;

    ++gcount;
}

std::string Graduate::research() const
{
    return gresearch;
}

std::string Graduate::advisor() const
{
    return gadvisor;
}

int Graduate::count()
{
    return gcount;
}

void Graduate::print_details() const
{
    Student::print_details();
    std::cout << "Type = Graduate Student" << std::endl;
    std::cout << "Research Area = " << gresearch << std::endl;
    std::cout << "Advisor = " << gadvisor << std::endl;
}

double Graduate::total_score() const
{
    double hw_mean_score = hw_mean();
    double proj_score = project_score();
    double total{};

    total = (hw_mean_score * 0.5) + (proj_score * 0.5);
    return total;
}

std::string Graduate::course_grade() const
{
    std::string letter_grade;
    double total = total_score();

    return letter_grade = (total >= 80.0) ? "CR" : "N";
}