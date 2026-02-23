/*!************************************************************************
  \file student.cpp

  \author Bryan Ang Wei Ze

  \par DP email: bryanweize.ang\@digipen.edu

  \par Course: CSD2126 Modern C++ Design Patterns

  \par Programming Assignment #4

  \date 26-10-2024
  
  \brief
  This source file contains definition of class Student member functions
**************************************************************************/
#include "student.hpp"
#include <iostream>
#include <sstream>

int Student::scount = 0;

Student::Student(std::string& line)
{
    size_t stud_name_end{}, year_end{}, proj_end{}, hw_end{};

    ++scount;
    sid = scount;

    // extract student name
    stud_name_end = line.find_first_of(',');
    sname = line.substr(0, stud_name_end);

    // remove student name portion from line
    line = line.substr(stud_name_end + 2);

    // extract studen yob
    year_end = line.find_first_of(',');
    syob = std::stoi(line.substr(0, year_end));

    // remove student yob portion from line
    line = line.substr(year_end + 2);

    // extract project grade
    proj_end = line.find_first_of(' ');
    sproj = std::stod(line.substr(0, proj_end));

    // remove project grade portion from line
    line = line.substr(proj_end + 1);

    // extract hw grades
    hw_end = line.find_first_of(',');
    std::istringstream hw_grades{line.substr(0, hw_end)};
    double grade{};
    while (hw_grades.good())
    {
        hw_grades >> grade;
        shw.push_back(grade);
    }
    
    // remove hw grades portion from line
    line = line.substr(hw_end + 2);
}

double Student::hw_mean() const
{
    double avg{};
    for (double grade : shw)
    {
        avg += grade;
    }
    avg /= shw.size();
    return avg;
}

int Student::id() const
{
    return sid;
}

std::string Student::name() const
{
    return sname;
}

int Student::yob() const
{
    return syob;
}

double Student::project_score() const
{
    return sproj;
}

int Student::count()
{
    return scount;
}

void Student::print_details() const
{
    std::cout << "Id = " << sid << std::endl;
    std::cout << "Name = " << sname << std::endl;
    std::cout << "Age = " << age() << std::endl;
    std::cout << "Project = " << sproj << std::endl;
    std::cout << "Assignment = [";
    for (size_t i = 0; i < shw.size(); ++i)
    {
        std::cout << shw[i];
        // if not the last assignment grade, print ", " after grade
        if (i != shw.size() - 1)
        {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
    std::cout << "Total = " << total_score() << std::endl;
    std::cout << "Grade = " << course_grade() << std::endl;
}