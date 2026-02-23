/*!************************************************************************
  \file process.cpp

  \author Bryan Ang Wei Ze

  \par DP email: bryanweize.ang\@digipen.edu

  \par Course: CSD2126 Modern C++ Design Patterns

  \par Programming Assignment #4

  \date 26-10-2024
  
  \brief
  This source file contains definition of functions that perform some
  processing on student objects.
**************************************************************************/
#include "process.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>

static bool compare_score(const Student* const & lhs, const Student* const & rhs)
{
  return lhs->total_score() > rhs->total_score();
}

void parse_file(std::ifstream &ifs, std::vector<Student*>& vs,
                std::vector<Student*>& vus, std::vector<Student*>& vgs)
{
  std::string str;
  while (getline(ifs, str))
  {
    char ch = str[0];
    str = str.substr(3);
    if (ch == 'U')
    {
      Student *ps = new Undergraduate(str);
      vs.push_back(ps);
      vus.push_back(ps);
    }
    else if (ch == 'G')
    {
      Student *ps = new Graduate(str);
      vs.push_back(ps);
      vgs.push_back(ps);
    }
  }
}

void print_records(std::vector<Student*> const& v)
{
  for (const Student* s : v)
  {
    s->print_details();
    std::cout << '\n';
  }
}

void print_stats(std::vector<Student*> const& v) {
  std::cout << "Number of students = " << v.size() << '\n';

  double total_score{};
  for (const Student* const& s : v)
  {
    total_score += s->total_score();
  }
  std::cout << "The mean of the total score = "<< total_score / v.size() << '\n';

  std::cout << "The sorted list of students (id, name, total, grade) in descending order of total:\n";

  std::vector<Student*> copy_v{v};
  std::sort(copy_v.begin(), copy_v.end(), compare_score);
  for (const Student* const& s : copy_v)
  {
    std::cout << s->id() << ", " << s->name() << ", " << s->total_score() << 
    ", "<< s->course_grade() << '\n';
  }
  std::cout << '\n';
}

/**
* @brief Return memory allocated by operator new ONLY using pointers contained
* in container specified by first parameter. Why? Because containers specified
* in the next two containers reference the same memory locations as elements of
* first container, double deletion will cause undefined run-time behavior.
* Next, clear contents of all 3 containers.
*
* @param vs Reference to vector<Student*> containing pointers to all students.
* @param vs Reference to vector<Student*> containing pointers to undergraduates.
* @param vs Reference to vector<Student*> containing pointers to graduates.
*/
void cleanup(std::vector<Student*>& vs, std::vector<Student*>& vus, std::vector<Student*>& vgs)
{
  // delete memory previously allocated by operator new
  for (Student *ps : vs)
  {
    delete ps;
  }
  // avoid double deletion!!!
  vs.clear();
  vus.clear();
  vgs.clear();
}
