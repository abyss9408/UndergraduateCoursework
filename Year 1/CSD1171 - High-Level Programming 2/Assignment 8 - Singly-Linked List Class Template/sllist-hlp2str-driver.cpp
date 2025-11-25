#include <cstddef>    // size_t
#include <iostream>   // std::cout
#include <array>      // std::array
#include <string>     // std::string and std::to_string
#include "sllist.hpp" // hlp2::sllist
#include "Str.hpp"    // hlp2::Str

// Note: compile with -D=DEBUG to ensure your output matches the output files!!!

namespace {
  void TestConstructorsAndSwap(); // 1
  void TestPushFrontBackAndPop(); // 2
  void TestAssignments();         // 3
  void TestAdditions();           // 4
  void TestSubscript();           // 5

  template <typename T> void InUse();
} // end anonymous namespace 

int main(int argc, char *argv[]) {
  if (argc < 2) {
		std::cout << "Usage: ./listint.out test-number\n";
		std::cout << "Tests 1 through 5 are specified by functions listed above.\n"
							<< "Test 0 will include all the 5 tests described above.\n";
    return 0;
  }

  int const num_of_tests {5};
  int test_num{std::stoi(argv[1])};
	test_num = test_num < 0 ? 0 : test_num;
	test_num = test_num > num_of_tests ? num_of_tests : test_num;

  using TPF = void (*)();
  std::array<TPF, num_of_tests> Tests {
    TestConstructorsAndSwap,  // 1
    TestPushFrontBackAndPop, // 2
    TestAssignments,         // 3
    TestAdditions,           // 4
    TestSubscript,           // 5
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
void TestConstructorsAndSwap() {
  std::cout << "---------------------TestConstructorsAndSwap---------------------\n";
  InUse<hlp2::Str>();

  std::cout << "testing default ctor, empty(), and size() ...\n";
  hlp2::sllist<hlp2::Str> list1;
  std::cout << "list1 is " << (list1.empty() ? "" : "not ") << "empty\n";
  std::cout << "Items in list1: " << list1.size() << "\n";
  InUse<hlp2::Str>();

  std::cout << "testing single-argument ctor, empty(), and size() ...\n";
  hlp2::sllist<hlp2::Str> list2({"4", "7", "12", "5", "9", "23", "7", "11", "15", "2"});
  std::cout << "list2: " << list2;
  std::cout << "list2 is " << (list2.empty() ? "" : "not ") << "empty\n";
  std::cout << "Items in list1/list2: " << list1.size() << "/" << list2.size() << "\n";
  InUse<hlp2::Str>();

  std::cout << "testing single-argument ctor, empty(), and size() ...\n";
  hlp2::sllist<hlp2::Str> list3{"11"};
  std::cout << "list3: " << list3;
  std::cout << "list3 is " << (list3.empty() ? "" : "not ") << "empty\n";
  std::cout << "Items in list1/list2/list3: " << list1.size() << "/"
            << list2.size() << "/" << list3.size() << "\n";
  InUse<hlp2::Str>();

  std::cout << "testing copy ctor, empty(), and size() ...\n";
  hlp2::sllist<hlp2::Str> list4{list2}; // copy construction
  std::cout << "list4: " << list4;
  std::cout << "list4 is " << (list4.empty() ? "" : "not ") << "empty\n";
  std::cout << "Items in list1/list2/list3/list4: " << list1.size() << "/"
            << list2.size() << "/" << list3.size() << "/" << list4.size() << "\n";
  InUse<hlp2::Str>();

  std::cout << "testing member function swap ...\n";
  std::cout << "Before swapping list3 and list4:\n";
  std::cout << "list3: " << list3;
  std::cout << "list4: " << list4;
  std::cout << "Items in list1/list2/list3/list4: " << list1.size() << "/"
            << list2.size() << "/" << list3.size() << "/" << list4.size() << "\n";
  InUse<hlp2::Str>();
  list3.swap(list4);
  std::cout << "After swapping list3 and list4:\n";
  std::cout << "list3: " << list3;
  std::cout << "list4: " << list4;
  std::cout << "Items in list1/list2/list3/list4: " << list1.size() << "/"
            << list2.size() << "/" << list3.size() << "/" << list4.size() << "\n";
  InUse<hlp2::Str>();

  std::cout << "testing non-member function swap ...\n";
  list3 = {"11"};
  list4 = list2;
  std::cout << "Before swapping list3 and list4:\n";
  std::cout << "list3: " << list3;
  std::cout << "list4: " << list4;
  std::cout << "Items in list1/list2/list3/list4: " << list1.size() << "/"
            << list2.size() << "/" << list3.size() << "/" << list4.size() << "\n";
  InUse<hlp2::Str>();
  hlp2::swap(list3, list4);
  std::cout << "After swapping list3 and list4:\n";
  std::cout << "list3: " << list3;
  std::cout << "list4: " << list4;
  std::cout << "Items in list1/list2/list3/list4: " << list1.size() << "/"
            << list2.size() << "/" << list3.size() << "/" << list4.size() << "\n";
  InUse<hlp2::Str>();

  std::cout << "testing array ctor, empty(), and size() ...\n";
  int const size10{10};
  std::array<hlp2::Str, size10> str_array{"-1","-2","-3","-4","-5","-6","-7","-8","-9","-10"};
  hlp2::sllist<hlp2::Str> list5(str_array.data(), str_array.data()+size10);
  std::cout << "list5: " << list5;
  std::cout << "list5 is " << (list5.empty() ? "" : "not ") << "empty\n";
  std::cout << "Items in list1/list2/list3/list4/list5: " << list1.size() << "/"
  << list2.size() << "/" << list3.size() << "/" << list4.size() << "/" << list5.size() << "\n";
  InUse<hlp2::Str>();
}

void TestPushFrontBackAndPop() {
  std::cout << "---------------------TestPushFrontBackAndPop---------------------\n";
  InUse<hlp2::Str>();

  std::cout << "testing push_front() and size() ...\n";
  int const size10{10};
  hlp2::sllist<hlp2::Str> list1;
  for (int i{1}; i <= size10; i++) {
    list1.push_front(hlp2::Str(std::to_string(-2*i).c_str()));
  }
  std::cout << "list1: " << list1;
  std::cout << "Items in list1: " << list1.size() << "\n";
  InUse<hlp2::Str>();

  std::cout << "testing push_back() and size() ...\n";
  hlp2::sllist<hlp2::Str> list2;
  for (int i{1}; i <= size10; i++) {
    list2.push_back(hlp2::Str(std::to_string(i).c_str()));
  }
  std::cout << "list2: " << list2;
  std::cout << "Items in list1/list2: " << list1.size() << "/" << list2.size() << "\n";
  InUse<hlp2::Str>();

  std::cout << "testing front() and size() ...\n";
  hlp2::Str &front_value = list2.front();
  front_value = std::to_string(-10).c_str();
  std::cout << "testing front() const ...\n";
  hlp2::Str const& const_front_value = list2.front();
  std::cout << "front value: " << const_front_value << "\n";
  std::cout << "list2: " << list2;
  std::cout << "Items in list1/list2: " << list1.size() << "/" << list2.size() << "\n";
  InUse<hlp2::Str>();

  std::cout << "testing both push_front(), push_back(), and size() ...\n";
  hlp2::sllist<hlp2::Str> list3;
  for (int i{1}; i <= size10; i++) {
    list3.push_front(std::to_string(i).c_str());
    list3.push_back(std::to_string(i * 5).c_str());
  }
  std::cout << "list3: " << list3;
  std::cout << "Items in list1/list2/list3: " << list1.size() << "/" << list2.size() 
            << "/" << list3.size() << "\n";
  InUse<hlp2::Str>();

  std::cout << "testing pop_front(), push_front(), front(), empty(), size() ...\n";
  hlp2::sllist<hlp2::Str> list4;
  for (int i{1}; i <= size10; i++) {
    list4.push_front(std::to_string(i).c_str());
  }
  std::cout << "Original list4: " << list4;
  std::cout << "Items in list1/list2/list3/list4: " << list1.size() << "/"
            << list2.size() << "/" << list3.size() << "/" << list4.size() << "\n";
  InUse<hlp2::Str>();
  while (!list4.empty()) {
    hlp2::Str item = list4.front();
    list4.pop_front();
    std::cout << "First item was: " << item << "\n";
    std::cout << "New list4: " << list4;
    std::cout << "Items in list4: " << list4.size() << "\n";
  }

  std::cout << "\nItems in list1/list2/list3/list4: " << list1.size() << "/"
            << list2.size() << "/" << list3.size() << "/" << list4.size() << "\n";
  InUse<hlp2::Str>();
}

void TestAssignments() {
  std::cout << "---------------------TestAssignments---------------------\n";
  InUse<hlp2::Str>();

  int const size10{10};
  hlp2::sllist<hlp2::Str> list1, list2;
  for (int i{1}; i <= size10; i++) {
    list1.push_front(std::to_string(i * 2).c_str());
  }
  std::cout << "testing copy assignment:" << "\n";
  std::cout << "list1: " << list1;
  std::cout << "list2: " << list2;
  std::cout << "Items in list1/list2: " << list1.size() << "/" << list2.size() << "\n";
  InUse<hlp2::Str>();

  list2 = list1;
  std::cout << "After assignment [list2 = list1]:" << "\n";
  std::cout << "list1: " << list1;
  std::cout << "list2: " << list2;
  std::cout << "Items in list1/list2: " << list1.size() << "/" << list2.size() << "\n";
  InUse<hlp2::Str>();

  hlp2::sllist<hlp2::Str> list3;
  std::cout << "list3: " << list3;
  list3 = list2 = list1;
  std::cout << "After assignment [list3 = list2 = list1]:" << "\n";
  std::cout << "list1: " << list1;
  std::cout << "list2: " << list2;
  std::cout << "list3: " << list3;
  std::cout << "Items in list1/list2/list3: " << list1.size() << "/" 
            << list2.size() << "/" << list3.size() << "\n";
  InUse<hlp2::Str>();

  std::cout << "testing copy assignment with initializer_list [list3 = list2 = list1]:" << "\n";
  std::cout << "Assigning initializer list: {\"4\",\"7\",\"12\",\"5\",\"9\",\"23\",\"7\",\"11\",\"15\",\"2\"}\n";
  list3 = list2 = list1 = {"4","7","12","5","9","23","7","11","15","2"};
  std::cout << "After assignment:" << "\n";
  std::cout << "list1: " << list1;
  std::cout << "list2: " << list2;
  std::cout << "list3: " << list3;

  std::cout << "Items in list1/list2/list3: " << list1.size() << "/"
            << list2.size() << "/" << list3.size() << "\n";
  InUse<hlp2::Str>();
}

void TestAdditions() {
  std::cout << "---------------------TestAdditions---------------------\n";
  InUse<hlp2::Str>();

  std::cout << "testing op+ and op+= [thro' op+] ...\n";
  int const size5{5};
  hlp2::sllist<hlp2::Str> list1, list2;
  for (int i{1}; i <= size5; i++) {
    list1.push_front(std::to_string(i).c_str());
    list2.push_front(std::to_string(i*10).c_str());
  }
  std::cout << "list1: " << list1;
  std::cout << "list2: " << list2;
  std::cout << "Items in list1/list2: " << list1.size() << "/" << list2.size() << "\n";
  InUse<hlp2::Str>();

  std::cout << "testing list1 + list2:\n";
  hlp2::sllist<hlp2::Str> const list3{list1 + list2};
  std::cout << "list1: " << list1;
  std::cout << "list2: " << list2;
  std::cout << "list3: " << list3;
  std::cout << "Items in list1/list2/list3: " << list1.size() << "/"
            << list2.size() << "/" << list3.size() << "\n";
  InUse<hlp2::Str>();

  std::cout << "testing list1 + list2 + list3:\n";
  hlp2::sllist<hlp2::Str> list4;
  list4 = list1 + list2 + list3;
  std::cout << "list4: " << list4;
  std::cout << "Items in list1/list2/list3/list4: " << list1.size() << "/"
            << list2.size() << "/" << list3.size() << "/" << list4.size() << "\n";
  InUse<hlp2::Str>();

  std::cout << "testing list4 += {\"-1\",\"4\",\"-5\",\"6\",\"-7\"}+list1+list2+list3+{\"31\",\"41\",\"51\",\"61\",\"71\"}:\n";
  list4 += std::initializer_list<hlp2::Str>{"-1","4","-5","6","-7"} + list1 + list2
                        + std::initializer_list<hlp2::Str>{"31","41","51","61","71"};
  std::cout << "list4: " << list4;
  std::cout << "Items in list1/list2/list3/list4: " << list1.size() << "/"
            << list2.size() << "/" << list3.size() << "/" << list4.size() << "\n";
  InUse<hlp2::Str>();

  std::cout << "testing list1 += list2 ...\n";
  std::cout << "list1: " << list1;
  std::cout << "list2: " << list2;
  list1 += list2;
  std::cout << "After addition list1 += list2:" << "\n";
  std::cout << "list1: " << list1;
  std::cout << "Items in list1/list2/list3/list4: " << list1.size() << "/"
            << list2.size() << "/" << list3.size() << "/" << list4.size() << "\n";
  InUse<hlp2::Str>();

  hlp2::sllist<hlp2::Str> list5;
  std::cout << "testing list1 += list5 [empty list] ...\n";
  std::cout << "list1: " << list1;
  std::cout << "list5: " << list5;
  list1 += list5;
  std::cout << "After addition list1 += list5 [empty list] ...\n";
  std::cout << "list1: " << list1;
  std::cout << "Items in list1/list2/list3/list4/list5: " << list1.size() << "/" << list2.size() << "/"
            << list3.size() << "/" << list4.size() << "/" << list5.size() << "\n";
  InUse<hlp2::Str>();

  hlp2::sllist<hlp2::Str> list6, list7, list8;
  for (int i{1}; i <= size5; i++) {
    list6.push_front(std::to_string(i).c_str());
    list7.push_front(std::to_string(i*2).c_str());
    list8.push_front(std::to_string(i*5).c_str());
  }

  std::cout << "testing list6 += list7 += list8 ...\n";
  std::cout << "list6: " << list6;
  std::cout << "list7: " << list7;
  std::cout << "list8: " << list8;
  list6 += list7 += list8;
  std::cout << "After addition list6 += list7 += list8 ...\n";
  std::cout << "list6: " << list6;
  std::cout << "list7: " << list7;
  std::cout << "list8: " << list8;

  std::cout << "Items in list1/list2/list3/list4/list5/list6/list7/list8: "
            << list1.size() << "/" << list2.size() << "/" << list3.size() << "/"
            << list4.size() << "/" << list5.size() << "/" << list6.size() << "/"
            << list7.size() << "/" << list8.size() << "\n";
  InUse<hlp2::Str>();
}

void TestSubscript() {
  std::cout << "---------------------TestSubscript---------------------\n";
  InUse<hlp2::Str>();

  std::cout << "testing op[] const ...\n";
  hlp2::sllist<hlp2::Str> const list1 {"-11","11","22","33","44","55","66","77","88","99"};
  std::cout << "list1: " << list1;
  std::cout << "list1[0] = " << list1[0] << "\n";
  std::cout << "list1[5] = " << list1[5] << "\n";
  std::cout << "list1[9] = " << list1[9] << "\n\n";
  std::cout << "Items in list1: " << list1.size() << "\n";
  InUse<hlp2::Str>();

  std::cout << "testing op[] ...\n";
  hlp2::sllist<hlp2::Str> list2{list1};
  std::cout << "list2: " << list2;
  std::cout << "list2[0] = \"10\":   " << (list2[0] = "10") << "\n";
  std::cout << "list2[5] = \"20\":   " << (list2[5] = "20") << "\n";
  std::cout << "list2[9] = \"30\":   " << (list2[9] = "30") << "\n";
  std::cout << "list2[7] = list2[2]: " << (list2[7] = list2[2]) << "\n";
  std::cout << "list2: " << list2;

  std::cout << "Items in list1/list2: " << list1.size() << "/" << list2.size() << "\n";
  InUse<hlp2::Str>();
}

template <typename T>
void InUse() {
  std::cout << "============================================\n";
  std::cout << "Total number of Lists/Nodes in use: ";
  std::cout << hlp2::sllist<T>::object_count() << "/" << hlp2::sllist<T>::node_count();
  std::cout << "\n============================================\n\n";
}

} // end anonymous namespace