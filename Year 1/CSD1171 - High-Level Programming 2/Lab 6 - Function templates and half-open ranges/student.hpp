#ifndef STUDENT_HPP
#define STUDENT_HPP

struct Student {
  std::string login;
  int age;
  int year;
  double GPA;
};

std::ostream& operator<<(std::ostream& os, Student const& s) {
  os << "(login) " << s.login << ", " << "(age) "   << s.age   << ", "
     << "(year) "  << s.year  << ", " << "(GPA) "   << s.GPA   << "\n";
  return os;
}

bool operator<(Student const& lhs, Student const& rhs) {
  return lhs.GPA < rhs.GPA;
}

void print(Student const *first, Student const *last) {
  while (first != last) {
    std::cout << *first;
    ++first;
  }
}

#endif // end STUDENT_HPP
