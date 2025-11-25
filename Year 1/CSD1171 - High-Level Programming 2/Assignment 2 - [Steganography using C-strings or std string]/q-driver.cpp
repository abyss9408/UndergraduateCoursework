#include <iostream> // std::cout
#include <string>   // std::stoi
#include "q.hpp"

// declarations of unit tests ...
namespace {
  void test0();
  void test1();
  void test2();
  void test3();
  void test4();
  void test5();
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: ./lab1.out test-number-between-0-and-4 > secret-message-file-name\n";
    return 0;
  }

  // stoi convert string such as "123" to integral value 123 ...
  int test_choice = std::stoi(argv[1]);
  test_choice = (test_choice < 0) ? 0 : (test_choice > 5) ? 5 : test_choice;
  switch (test_choice) {
    case 0: test0(); break;
    case 1: test1(); break;
    case 2: test2(); break;
    case 3: test3(); break;
    case 4: test4(); break;
    case 5: test5(); break;
  }
}

namespace {
// definitions of unit tests ...
void test0() {
  hlp2::extract("cover-message0.txt", nullptr);
}

void test1() {
  char const *keywords1[] = {"please", "shark", nullptr};
  hlp2::extract("cover-message1.txt", keywords1);
}

void test2() {
  char const *keywords2[] = {"so", "rain", "in", nullptr};
  hlp2::extract("cover-message2.txt", keywords2);
}

void test3() {
  char const *keywords3[] = {"are", "a", nullptr};
  hlp2::extract("cover-message3.txt", keywords3);
}

void test4() {
  char const *keywords4[] = {"is", "that", "better", nullptr};
  hlp2::extract("cover-message4.txt", keywords4);
}

void test5() {
  char const *keywords5[] = {"shark", "please", nullptr};
  hlp2::extract("cover-message5.txt", keywords5);
}
} // end anonymous namespace
