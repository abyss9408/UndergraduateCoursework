#include <iostream>
#include <initializer_list>
#include <array>
#include <string>
#include "listint.hpp"

namespace {
  void TestConstructors();        // 1
  void TestPushFrontBackAndPop(); // 2
  void TestAssignments();         // 3
  void TestAdditions();           // 4
  void TestSubscript();           // 5
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
		std::cout << "Usage: ./listint.out test-number\n";
		std::cout << "Tests 1 through 5 are specified by functions\n"
		          << "TestConstructors, TestPushFrontBackAndPop, TestAssignments,\n"
							<< "TestAdditions, and TestSubscript.\n"
							<< "Test 0 will include all the 5 tests described above.\n";
    return 0;
  }

  int const num_of_tests {5};
  int test_num{std::stoi(argv[1])};
	test_num = test_num < 0 ? 0 : test_num;
	test_num = test_num > num_of_tests ? num_of_tests : test_num;

  using TPF = void (*)();
  std::array<TPF, num_of_tests> Tests {
    TestConstructors,        // 1
    TestPushFrontBackAndPop, // 2
    TestAssignments,         // 3
    TestAdditions,           // 4
    TestSubscript            // 5
  };
  
  if (test_num == 0) {
		for (TPF x : Tests) {
      x();
    }
  } else {
    Tests[test_num - 1]();
  }
}

namespace {
void TestConstructors() {
  std::cout << "---------------------TestConstructors---------------------" << std::endl;

  std::cout << "============================================\n";
  std::cout << "Total number of Lists created: ";
  std::cout << hlp2::ListInt::object_count();
  std::cout << std::endl;
  std::cout << "============================================\n";
  std::cout << std::endl;

  std::cout << "testing default sctor ...\n";
  int const size10{10};
  hlp2::ListInt list1;
  std::cout << "list1 is " << (list1.empty() ? "" : "not ") << "empty\n";
  std::cout << "Items in list1: " << list1.size() << std::endl;

  std::cout << "use push_front to add 10 elements to list1\n";
  for (int i{1}; i <= size10; i++) {
    list1.push_front(i * 3);
  }
  std::cout << "list1: " << list1;
  std::cout << "Items in list1: " << list1.size() << std::endl;

  std::cout << "============================================\n";
  std::cout << "Total number of Lists created: ";
  std::cout << hlp2::ListInt::object_count();
  std::cout << std::endl;
  std::cout << "============================================\n";
  std::cout << std::endl;

  std::cout << "testing copy ctor ...\n";
  hlp2::ListInt list2{list1}; // copy construction
  std::cout << "list2: " << list2;
  std::cout << "list2 is " << (list2.empty() ? "" : "not ") << "empty\n";
  std::cout << "Items in list2: " << list2.size() << std::endl;

  std::cout << "============================================\n";
  std::cout << "Total number of Lists created: ";
  std::cout << hlp2::ListInt::object_count();
  std::cout << std::endl;
  std::cout << "============================================\n";
  std::cout << std::endl;
  
  std::cout << "testing single-argument ctor ...\n";
  hlp2::ListInt list3({4, 7, 12, 5, 9, 23, 7, 11, 15, 2});
  std::cout << "list3: " << list3;
  std::cout << "list3 is " << (list3.empty() ? "" : "not ") << "empty\n";
  std::cout << "Items in list3: " << list3.size() << std::endl;

  std::cout << "============================================\n";
  std::cout << "Total number of Lists created: ";
  std::cout << hlp2::ListInt::object_count();
  std::cout << std::endl;
  std::cout << "============================================\n";
  std::cout << std::endl;

  std::cout << "testing single-argument ctor ...\n";
  hlp2::ListInt list4{11};
  std::cout << "list4: " << list4;
  std::cout << "list4 is " << (list4.empty() ? "" : "not ") << "empty\n";
  std::cout << "Items in list4: " << list4.size() << std::endl;

  std::cout << "============================================\n";
  std::cout << "Total number of Lists created: ";
  std::cout << hlp2::ListInt::object_count();
  std::cout << std::endl;
  std::cout << "============================================\n";
  std::cout << std::endl;
}

void TestPushFrontBackAndPop() {
  std::cout << "---------------------TestPushFrontBackAndPop---------------------" << std::endl;

  std::cout << "============================================\n";
  std::cout << "Total number of Lists created: ";
  std::cout << hlp2::ListInt::object_count();
  std::cout << std::endl;
  std::cout << "============================================\n";
  std::cout << std::endl;

  std::cout << "testing ListInt::push_front() ...\n";
  int const size5{5};
  hlp2::ListInt list_pushfront;
  for (int i{1}; i <= size5; i++)
    list_pushfront.push_front(i);

  std::cout << list_pushfront;
  std::cout << "Items in the list: " << list_pushfront.size() << std::endl;
  std::cout << std::endl;

  std::cout << "============================================\n";
  std::cout << "Total number of Lists created: ";
  std::cout << hlp2::ListInt::object_count();
  std::cout << std::endl;
  std::cout << "============================================\n";
  std::cout << std::endl;

  std::cout << "testing ListInt::push_back() ...\n";
  hlp2::ListInt list_pushback;
  for (int i{1}; i <= size5; i++)
    list_pushback.push_back(i);

  std::cout << list_pushback;
  std::cout << "Items in the list: " << list_pushback.size() << std::endl;
  std::cout << std::endl;

  std::cout << "============================================\n";
  std::cout << "Total number of Lists created: ";
  std::cout << hlp2::ListInt::object_count();
  std::cout << std::endl;
  std::cout << "============================================\n";
  std::cout << std::endl;

  std::cout << "testing ListInt::push_front() and ListInt::push_back() ...\n";
  int const size10{10};
  hlp2::ListInt list_pushfrontback;
  for (int i{1}; i <= size10; i++) {
    list_pushfrontback.push_front(i);
    list_pushfrontback.push_back(i * 5);
  }

  std::cout << list_pushfrontback;
  std::cout << "Items in the list: " << list_pushfrontback.size() << std::endl;
  std::cout << std::endl;

  std::cout << "============================================\n";
  std::cout << "Total number of Lists created: ";
  std::cout << hlp2::ListInt::object_count();
  std::cout << std::endl;
  std::cout << "============================================\n";
  std::cout << std::endl;

  std::cout << "testing ListInt::pop_front() ...\n";
  hlp2::ListInt list_popfront;
  for (int i{1}; i <= size10; i++) {
    list_popfront.push_front(i);
  }

  std::cout << "Original list: " << list_popfront;
  std::cout << "Items in the list: " << list_popfront.size() << std::endl;
  while (!list_popfront.empty()) {
    int item = list_popfront.pop_front();
    std::cout << "First item was: " << item << std::endl;
    std::cout << "New list: " << list_popfront;
    std::cout << "Items in the list: " << list_popfront.size() << std::endl;
  }
  std::cout << std::endl;

  std::cout << "============================================\n";
  std::cout << "Total number of Lists created: ";
  std::cout << hlp2::ListInt::object_count();
  std::cout << std::endl;
  std::cout << "============================================\n";
  std::cout << std::endl;
}

void TestAssignments() {
  std::cout << "---------------------TestAssignments---------------------" << std::endl;

  std::cout << "============================================\n";
  std::cout << "Total number of Lists created: ";
  std::cout << hlp2::ListInt::object_count();
  std::cout << std::endl;
  std::cout << "============================================\n";
  std::cout << std::endl;

  int const size{10};
  hlp2::ListInt list1, list2;
  for (int i = 1; i <= size; i++) {
    list1.push_front(i * 2);
  }
  std::cout << "list1: " << list1;
  std::cout << "Before assignment:" << std::endl;
  std::cout << "list2: " << list2;

  list2 = list1;

  std::cout << "After assignment:" << std::endl;
  std::cout << "list1: " << list1;
  std::cout << "list2: " << list2;

  hlp2::ListInt list3;
  std::cout << "Before assignment:" << std::endl;
  std::cout << "list3: " << list3;
  list3 = list2 = list1;
  std::cout << "After assignment:" << std::endl;
  std::cout << "list1: " << list1;
  std::cout << "list2: " << list2;
  std::cout << "list3: " << list3;

  std::cout << "\nAssigning initializer list: {4, 7, 12, 5, 9, 23, 7, 11, 15, 2}\n";
  list3 = list2 = list1 = {4, 7, 12, 5, 9, 23, 7, 11, 15, 2};
  std::cout << "After assignment:" << std::endl;
  std::cout << "list1: " << list1;
  std::cout << "list2: " << list2;
  std::cout << "list3: " << list3 << std::endl;

  std::cout << "============================================\n";
  std::cout << "Total number of Lists created: ";
  std::cout << hlp2::ListInt::object_count();
  std::cout << std::endl;
  std::cout << "============================================\n";
  std::cout << std::endl;
}

void TestAdditions() {
  std::cout << "---------------------TestAdditions---------------------" << std::endl;

  std::cout << "============================================\n";
  std::cout << "Total number of Lists created: ";
  std::cout << hlp2::ListInt::object_count();
  std::cout << std::endl;
  std::cout << "============================================\n";
  std::cout << std::endl;

  std::cout << "testing operator+ ...\n";
  int const size5{5};
  hlp2::ListInt list1, list2;
  for (int i{1}; i <= size5; i++) {
    list1.push_front(i);
    list2.push_front(i * 10);
  }
  std::cout << "Before addition:" << std::endl;
  std::cout << "list1: " << list1;
  std::cout << "list2: " << list2;

  hlp2::ListInt const list3{list1 + list2};
  std::cout << "After addition list1+list2:" << std::endl;
  std::cout << "list3: " << list3;

  hlp2::ListInt list4;
  list4 = list1 + list2 + list3;
  std::cout << "After addition list1+list2+list3:" << std::endl;
  std::cout << "list4: " << list4;
  std::cout << "Items in the list: " << list4.size() << std::endl;
  std::cout << std::endl;

  list4 = list1 + list2 + std::initializer_list<int>({31, 41, 51, 61, 71});
  std::cout << "After addition list1+list2+{31, 41, 51, 61, 71}:" << std::endl;
  std::cout << "list4: " << list4;
  std::cout << "Items in the list: " << list4.size() << std::endl;
  std::cout << std::endl;

  list4 = std::initializer_list<int>({31, 41, 51, 61, 71}) + list1 + list2;
  std::cout << "After addition {31, 41, 51, 61, 71}+list1+list2:" << std::endl;
  std::cout << "list4: " << list4;
  std::cout << "Items in the list: " << list4.size() << std::endl;
  std::cout << std::endl;

  std::cout << "testing ListInt::operator+= ...\n";

  std::cout << "Before addition:" << std::endl;
  std::cout << "list1: " << list1;
  std::cout << "list2: " << list2;

  list1 += list2;
  std::cout << "After addition list1 += list2:" << std::endl;
  std::cout << "list1: " << list1;
  std::cout << "list2: " << list2;
  std::cout << "Items in list1: " << list1.size() << std::endl;
  std::cout << std::endl;

  hlp2::ListInt list5;
  std::cout << "Before addition:" << std::endl;
  std::cout << "list1: " << list1;
  std::cout << "list5: " << list5;

  list1 += list5;
  std::cout << "After addition list1 += list5:" << std::endl;
  std::cout << "list1: " << list1;
  std::cout << "list5: " << list5;
  std::cout << "Items in list1: " << list1.size() << std::endl;
  std::cout << std::endl;

  hlp2::ListInt list6, list7, list8;
  for (int i{1}; i <= size5; i++) {
    list6.push_front(i);
    list7.push_front(i * 2);
    list8.push_front(i * 5);
  }

  std::cout << "Before addition:" << std::endl;
  std::cout << "list6: " << list6;
  std::cout << "list7: " << list7;
  std::cout << "list8: " << list8;

  list6 += list7 += list8;
  std::cout << "After addition:" << std::endl;
  std::cout << "list6: " << list6;
  std::cout << "list7: " << list7;
  std::cout << "list8: " << list8;
  std::cout << "Items in the list: " << list6.size() << std::endl;
  std::cout << std::endl;

  std::cout << "============================================\n";
  std::cout << "Total number of Lists created: ";
  std::cout << hlp2::ListInt::object_count();
  std::cout << std::endl;
  std::cout << "============================================\n";
  std::cout << std::endl;
}

void TestSubscript() {
  std::cout << "---------------------TestSubscript---------------------" << std::endl;

  std::cout << "============================================\n";
  std::cout << "Total number of Lists created: ";
  std::cout << hlp2::ListInt::object_count();
  std::cout << std::endl;
  std::cout << "============================================\n";
  std::cout << std::endl;

  std::cout << "testing operator[](size_type) const\n";

  hlp2::ListInt const list1 {-11, 11, 22, 33, 44, 55, 66, 77, 88, 99 };
  std::cout << "list1: " << list1;

  std::cout << "list[0] = " << list1[0] << std::endl;
  std::cout << "list[5] = " << list1[5] << std::endl;
  std::cout << "list[9] = " << list1[9] << std::endl;
  std::cout << std::endl;

  std::cout << "testing operator[](size_type) \n";

  hlp2::ListInt list2{list1};
  std::cout << "Original list2: " << list2;

  list2[0] = 10;
  list2[5] = 20;
  list2[9] = 30;
  list2[7] = list2[2];

  std::cout << "Modified list2: " << list2;
  std::cout << std::endl;
}
} // end anonymous namespace
