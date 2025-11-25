#include "vector.hpp" // vector methods
#include <iostream>
#include <string>
#include <array>
#include <cstdlib>  // atoi

namespace {
  void TestPush();
  void TestSubscripts();
  void TestALot1();
  void TestCopy();
  void TestAssign();
  void Print(hlp2::vector const& array, std::string const& newline = "\n");
} // end anonymous namespace

int main(int argc, char **argv) {
  int test_num{};
  if (argc > 1) {
    test_num = std::atoi(argv[1]);
  }

  using TPF = void (*)();
  int const num_of_tests {5};
  std::array<TPF, num_of_tests> Tests = {
    TestPush,        // 1 
    TestSubscripts,  // 2
    TestCopy,        // 3
    TestAssign,      // 4
    TestALot1        // 5
  };

  if (test_num == 0) {
    for (int i = 0; i < num_of_tests; i++) {
      Tests[i]();
    }
  } else if (test_num > 0 && test_num <= num_of_tests) {
    Tests[test_num - 1]();
  }
}

namespace {
void TestPush() {
  std::cout << "\n********** TestPush **********\n";
  hlp2::vector a;
  std::cout << "Empty array:\n";
  Print(a);

  std::cout << "add 5 integers:\n";
  a = {99, 98, 97, 96, 95};
  Print(a);

  std::cout << "push_back 5 integers:\n";
  for (hlp2::vector::value_type i{}; i < 5; i++) {
    a.push_back(i);
    Print(a);
  }

  std::cout << "push_back 5 more integers:\n";
  for (hlp2::vector::value_type i{}; i < 5; i++) {
    a.push_back(-i);
    Print(a);
  }
}

void TestSubscripts() {
  std::cout << "\n********** TestSubscripts **********\n";
  hlp2::vector a{11,12,13,14,15};
  for (int& x : a) {
    x *= 2;
  }
  Print(a);

  hlp2::vector::value_type const max_cnt {10};
  std::cout << "push_back 10 even integers:\n";
  for (hlp2::vector::value_type i{}; i < max_cnt; i++) {
    a.push_back(2 * i);
  }

  Print(a);

  std::cout << "multiply each value by 3:\n";
  for (hlp2::vector::value_type i{}; i < max_cnt; i++) {
    a[i] *= 3;
  }
  Print(a);

  std::cout << "use subscript with non-default constructed vector:\n";
  hlp2::vector b(10);
  for (hlp2::vector::size_type i{}; i < b.size(); ++i) {
    b[i] = 3*i;
  }
  Print(b);

  std::cout << "push back 10 odd ints:\n";
  hlp2::vector c;
  for (int i{}; i < 10; ++i) {
    c.push_back(-i*2+3);
  }
  Print(c);

  hlp2::vector::size_type index {c.size() / 2};
  std::cout << "using subscript: c[" << index << "]" << std::endl;
  std::cout << "c[" << index << "] = " << c[index] << std::endl;

  const hlp2::vector d{c};
  std::cout << "subscript operator for const objects:\n";
  for (hlp2::vector::size_type i{}; i < c.size(); ++i) {
    c[i] = d[i]-1;
  }
  Print(c);
}

void TestCopy() {
  std::cout << "\n********** TestCopy **********\n";
  hlp2::vector a;

  std::cout << "push_back 10 even integers:\n";
  for (int i{}; i < 10; i++) {
    a.push_back(2 * i);
  }

  std::cout << "Copy: b(a), print a,b\n";
  hlp2::vector b{a};
  Print(a);
  Print(b);

  std::cout << "Copy: c(b), print b,c\n";
  const hlp2::vector c{b};
  Print(b);
  Print(c);
}

void TestAssign() {
  std::cout << "\n********** TestAssign **********\n";
  hlp2::vector a, b, c;

  std::cout << "push_back 10 integers into a,b and 5 into c:\n";
  for (int i{}; i < 10; i++) {
    a.push_back(2 * i);
    b.push_back(i);
    if (i % 2) {
      c.push_back(i * 3);
    }
  }

  Print(a);
  Print(b);
  Print(c);

  std::cout << "Assign: b = a, print a,b\n";
  b = a;
  Print(a);
  Print(b);

  std::cout << "Assign: a = a, print a\n";
  a = a;
  Print(a);

  std::cout << "Assign: a = c, print a,c\n";
  a = c;
  Print(a);
  Print(c);

  std::cout << "Assign: c = b, print b,c\n";
  c = b;
  Print(b);
  Print(c);
}

void TestALot1() {
  std::cout << "\n********** TestALot1 **********\n";
  hlp2::vector a;

  std::cout << "push_back 10 even integers:\n";
  for (int i{}; i < 10; i++) {
    a.push_back(2 * i);
  }
  Print(a);

  std::cout << "push_back 10 odd integers:\n";
  for (int i{}; i < 10; i++) {
    a.push_back(2 * i + 1);
  }
  Print(a);

  std::cout << "empty/fill with 10 ints 3 times:\n";
  for (int i{}; i < 3; i++) {
    a.resize(0);

    for (int j{}; j < 10; j++) {
      a.push_back(j);
    }
  }
  Print(a);

  std::cout << "remove all but 3 integers:\n";
  a.resize(3);
  Print(a);

  std::cout << "push_back 8 more integers:\n";
  for (int i{}; i < 8; i++) {
    a.push_back(i);
  }
  Print(a);
}

void Print(hlp2::vector const& array, std::string const& newline) {
  for (int x : array) {
    std::cout << x << " ";
  }

  std::cout << "(size=" << array.size() << ", capacity=" << 
                           array.capacity() << ", allocs=" << 
                           array.allocations() << ")";
  std::cout << newline;
}

} // end anonymous namespace
