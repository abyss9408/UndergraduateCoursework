/*!************************************************************************
  \file ug.cpp

  \author Bryan Ang Wei Ze

  \par DP email: bryanweize.ang\@digipen.edu

  \par Course: CSD2126 Modern C++ Design Patterns

  \par Programming Assignment #4

  \date 26-10-2024
  
  \brief
  This source file contains definition of class Undergraduate member functions
**************************************************************************/
#include "ug.hpp"
#include <iostream>
#include <sstream>

int Undergraduate::ucount = 0;

Undergraduate::Undergraduate(std::string& line) : Student(line)
{
    size_t dorm_end{};

    // extract dorm name
    dorm_end = line.find_first_of(',');
    udorm = line.substr(0, dorm_end);

    // remove dorm portion from line
    line = line.substr(dorm_end + 2);

    // extract student year of study
    uyear = line;

    ++ucount;
}

std::string Undergraduate::dormitory() const
{
    return udorm;
}

std::string Undergraduate::year() const
{
    return uyear;
}

int Undergraduate::count()
{
    return ucount;
}

void Undergraduate::print_details() const
{
    Student::print_details();
    std::cout << "Type = Undergraduate Student" << std::endl;
    std::cout << "Residence Hall = " << udorm << std::endl;
    std::cout << "Year in College = " << uyear << std::endl;
}

double Undergraduate::total_score() const
{
    double hw_mean_score = hw_mean();
    double proj_score = project_score();
    double total{};

    total = (hw_mean_score * 0.7) + (proj_score * 0.3);
    return total;
}

std::string Undergraduate::course_grade() const
{
    std::string letter_grade;
    double total = total_score();

    return letter_grade = (total >= 70.0) ? "CR" : "N";
}