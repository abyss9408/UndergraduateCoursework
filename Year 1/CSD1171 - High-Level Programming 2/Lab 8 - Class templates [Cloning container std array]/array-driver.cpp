#include <iostream>
#include <iomanip>
#include <string>
#include <utility>
#include "array.hpp"

// private functions used in this source file ...
namespace {

// C-style function to return value obtained by subtracting 99 from parameter ...
int c_sub99(int ptr);

// C-style function that applies callback function to each array element ...
void c_xform(int *first, size_t N, int (*pf)(int));

// C-style function to return sum of elements of an int array
int c_sum(int const *first, size_t N);

// write to output stream a pair<T1,T2>
template <typename T1, typename T2>
std::ostream& operator<<(std::ostream& os, std::pair<T1, T2>const& rhs);

// can be called with pair of following iterators that define a range:
// cont.begin() and cont.end() OR 
// cont.begin() const and cont.end() const OR
// cont.cbegin() and cont.cend()
template <typename It>
void PRINT_ELEMENTS(It first, It second, std::string const& prefix="", 
                   std::string const& sep=" ", std::string const& postfix="\n");

// uses cont.cbegin() and cont.cend()
template <typename Cont>
void PRINT_ELEMENTS(Cont const& cont, std::string const& prefix="", 
                   std::string const& sep=" ", std::string const& postfix="\n");

template <typename Cont, size_t N>
void PRINT_ELEMENTS(Cont const& cont, std::string const& prefix="",
                  std::string const& sep=" ", std::string const& postfix="\n");

void test_with_string();
void test_with_int();
void test_with_pair();
} // end anonymous namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << "Usage: ./a.out test-number-between-0-and-3\n";
    return 0;
  }

  int const NUM_TESTS {3};
  int test_num = std::stoi(argv[1]);
  // sanity check ...
  test_num = test_num < 0 ? -test_num : test_num;
  // another sanity check ...
  test_num = (0 >= test_num && test_num <= NUM_TESTS) ? test_num : (test_num % (NUM_TESTS+1));

  using TPF = void (*)();
  TPF Tests[NUM_TESTS] {
    test_with_string, // 1 - correct output is in output file test1.txt
    test_with_int,    // 2 - correct output is in output file test2.txt
    test_with_pair    // 3 - correct output is in output file test3.txt
  };

  if (test_num == 0) { // correct output when argv[1] is "0" is in output file test0.txt
    for (TPF x : Tests) { x(); }
  } else {
    Tests[test_num - 1]();
  }
}

namespace {

void test_with_string() {
  std::cout << "------------------TEST_WITH_STRING---------------------------\n";

  using ARR_STR5 = hlp2::Array<std::string,5>;

  // Test: construction with initializer list
  ARR_STR5 a{"1","2","3","4","5"};
  static_assert(sizeof(a) == sizeof(std::string)*5, "hlp2::Array<T,N> must have zero overhead!!!");
  
  // Test: using aliases in class template hlp2::Array<T,N>

  // Test if value_type using alias was defined
  ARR_STR5::value_type value {a[0]};
  std::cout << "a[0]: " << value << '\n';
  
  // Test if size_type using alias was defined
  ARR_STR5::size_type size {a.size()};
  std::cout << "a has size: " << size << '\n';

  // Test if pointer using alias was defined
  ARR_STR5::pointer ptr {&a[0]};
  *ptr = "111";
  std::cout << "a[0]: " << a[0] << '\n';

  // Test copy construction works!!!
  ARR_STR5 const ca{a};

  // Test if const_pointer using alais was defined
  ARR_STR5::const_pointer cptr {&ca[ca.size()-1]};
  // *cptr = "555"; // should not compile!!!
  std::cout << "ca[" << (ca.size()-1) << "]: " << *cptr << '\n';

  // Test if reference using alais was defined
  ARR_STR5::reference ref {a[a.size()-1]};
  ref = "555";
  std::cout << "a[" << (a.size()-1) << "]: " << ref << '\n';

  // Test if const_reference using alais was defined
  ARR_STR5::const_reference cref {ca[ca.size()-1]};
  // cref = "55555"; // should not compile!!!
  std::cout << "ca[" << (ca.size()-1) << "]: " << cref << '\n';

  // Test iterator using alias and methods begin() and end()
  ARR_STR5::iterator first_it {a.begin()}; *first_it = "1111";
  ARR_STR5::iterator last_it  {a.end()};   *(last_it - 1) = "5555";
  PRINT_ELEMENTS(first_it, last_it, "a: ", " ");
  std::cout << "sizeof(a) == " << sizeof(a) << '\n';

  // Test copy assignment:
  a = ca;

  // Test const_iterator using alias and methods begin() and end()
  ARR_STR5::const_iterator first_cit {a.begin()}; // *first_cit = "1111";
  ARR_STR5::const_iterator last_cit  {a.end()};   // *(last_cit - 1) = "5555";
  PRINT_ELEMENTS(first_cit, last_cit, "a: ", " ");
  std::cout << "sizeof(a) == " << sizeof(a) << '\n';

  // Test methods: cbegin() and cend()
  first_cit = ca.begin();
  last_cit  = ca.end();
  PRINT_ELEMENTS(first_cit, last_cit, "ca: ", " ");
  std::cout << "sizeof(ca) == " << sizeof(ca) << '\n';

  // Test copy assignment with initializer list
  a = {"111", "222", "333", "444", "555"};
  PRINT_ELEMENTS(a.cbegin(), a.cend(), "a: ", " ");

  // Test method fill()
  a.fill("a");
  // Test: cbegin() and cend()
  PRINT_ELEMENTS(a, "a: ", " ");

  // Test method fill() again ...
  ARR_STR5 b; // std::string elements of built-in array are default constructed
  b.fill("z");

  // Test method operator[]()
  PRINT_ELEMENTS<ARR_STR5,5>(b, "b: ", " ");
  std::cout << "sizeof(b) == " << sizeof(b) << '\n';

  // Test method swap()
  a.swap(b);
  PRINT_ELEMENTS(a, "a: ", " ");
  PRINT_ELEMENTS(b, "b: ", " ");
	
  // Test copy assignment using initializer list
  a = {"today", "is", "a", "good", "day"};
  PRINT_ELEMENTS(a, "a: ", " ");
  b = {"Tomorrow", "will", "be", "much", "better"};
  PRINT_ELEMENTS(b, "b: ", " ");

  // Test methods: empty(), size(), front() and back()
  std::cout << std::boolalpha << b.empty() << '\n';
  if (!b.empty()) {
    b.front() = "front"; b.back() = "back";
    std::cout << "b: ";
    for (size_t i{}; i < b.size(); ++i) {
      std::cout << b[i] << " ";
    }
    std::cout << "\n";
  }

  // Test: copy constructor
  ARR_STR5 const cb = b;
  // Test method operator[]() const
  PRINT_ELEMENTS<ARR_STR5,5>(cb, "cb: ", " ");

  // Test non-member functions:
  // hlp2::swap(), hlp2::op==(), hlp2::op!=(), hlp2::op<(), hlp2::op>()
  std::cout << "Testing non-member swap:\n";
  ARR_STR5 a1{"a", "b", "c", "d", "e"}, a2{"a","b","c","d","d"}, a3{a2};
  PRINT_ELEMENTS(a1, "a1: ", " ");
  PRINT_ELEMENTS(a2, "a2: ", " ");
  hlp2::swap(a1, a2);
  PRINT_ELEMENTS(a1, "a1: ", " ");
  PRINT_ELEMENTS(a2, "a2: ", " ");

  std::cout << "Testing non-member swap:\n";
  hlp2::swap(a1, a2);
  PRINT_ELEMENTS(a1, "a1: ", " ");
  PRINT_ELEMENTS(a2, "a2: ", " ");
  PRINT_ELEMENTS(a3, "a3: ", " ");

  std::cout << "Testing non-member relational operators:\n";
  std::cout << "a1 " << (a1 < a2 ? "<" : a1>a2 ? ">" : "==") << " a2\n";
  std::cout << "a1 " << (a1 < a3 ? "<" : a1>a3 ? ">" : "==") << " a3\n";
  std::cout << "a1 " << (a1 != a2 ? "!=" : "==") << " a2\n";
  std::cout << "a1 " << (a1 != a3 ? "!=" : "==") << " a3\n";

  std::cout << "a2 " << (a2 < a1 ? "<" : a2>a1 ? ">" : "==") << " a1\n";
  std::cout << "a2 " << (a2 < a3 ? "<" : a2>a3 ? ">" : "==") << " a3\n";
  std::cout << "a2 " << (a2 != a1 ? "!=" : "==") << " a1\n";
  std::cout << "a2 " << (a2 != a3 ? "!=" : "==") << " a3\n";

  std::cout << "a3 " << (a3 < a1 ? "<" : a3>a1 ? ">" : "==") << " a1\n";
  std::cout << "a3 " << (a3 < a2 ? "<" : a3>a2 ? ">" : "==") << " a2\n";
  std::cout << "a3 " << (a3 != a1 ? "!=" : "==") << " a1\n";
  std::cout << "a3 " << (a3 != a2 ? "!=" : "==") << " a2\n";

  std::cout << "Testing non-member relational operators:\n";
  a1 = a2; a1[4] = "a";
  PRINT_ELEMENTS(a1, "a1: ", " ");
  PRINT_ELEMENTS(a2, "a2: ", " ");
  std::cout << "a1 " << (a1 < a2 ? "<" : a1>a2 ? ">" : "==") << " a2\n";
  std::cout << "a1 " << (a1 < a3 ? "<" : a1>a3 ? ">" : "==") << " a3\n";
}

void test_with_int() {
  std::cout << "------------------TEST_WITH_INT------------------------------\n";

  using ARR_INT10 = hlp2::Array<int,10>;

  // Test: default construction with initializer list
  ARR_INT10 a{1,3,5,7,9,11,13,15,17,19};
  // Test: begin() and end()
  PRINT_ELEMENTS(a.begin(), a.end(), "a: ", " ");
  std::cout << "sizeof(a) == " << sizeof(a) << '\n';
  // Test: fill()
  a.fill(99);
  // Test: cbegin() and cend()
  PRINT_ELEMENTS(a, "a: ", " ");

  // Test: fill()
  ARR_INT10 b; b.fill(86);
  // Test: operator[]()
  PRINT_ELEMENTS<ARR_INT10,10>(b, "b: ", " ");

  // Test: swap()
  a.swap(b);
  PRINT_ELEMENTS(a, "a: ", " ");
  PRINT_ELEMENTS(b, "b: ", " ");

  // Test: empty(), size(), front() and back()
  std::cout << std::boolalpha << b.empty() << '\n';
  if (!b.empty()) {
    b.front() = 0; b.back() = 100;
    std::cout << "b: ";
    for (size_t i{}; i < b.size(); ++i) {
      std::cout << b[i] << " ";
    }
    std::cout << "\n";
  }

  // Test: default assignment
  ARR_INT10 const cb = b;
  // Test: operator[]() const
  PRINT_ELEMENTS<ARR_INT10,10>(cb, "cb: ", " ");

  // Test method data() const
  std::cout << "Sum of b's elements: " << c_sum(b.data(), b.size()) << '\n';
  // Test method data() const again ...
  std::cout << "Sum of cb's elements: " << c_sum(cb.data(), cb.size()) << '\n';

  // Test method data()
  c_xform(b.data(), b.size(), c_sub99);
  std::cout << "Sum of b's elements: " << c_sum(b.data(), b.size()) << '\n';

  // Test non-member functions:
  // hlp2::swap(), hlp2::op==(), hlp2::op!=(), hlp2::op<(), hlp2::op>()
  using ARR_INT5 = hlp2::Array<int,5>;
  ARR_INT5 a1{-1, -2, -3, -4, -5}, a2{1, 2, 3, 4, 5}, a3{a2};
  PRINT_ELEMENTS(a1, "a1: ", " ");
  PRINT_ELEMENTS(a2, "a2: ", " ");
  std::cout << "Testing non-member swap:\n";
  hlp2::swap(a1, a2);
  PRINT_ELEMENTS(a1, "a1: ", " ");
  PRINT_ELEMENTS(a2, "a2: ", " ");

  std::cout << "Testing non-member swap:\n";
  hlp2::swap(a1, a2);
  PRINT_ELEMENTS(a1, "a1: ", " ");
  PRINT_ELEMENTS(a2, "a2: ", " ");
  PRINT_ELEMENTS(a3, "a3: ", " ");

  std::cout << "Testing non-member relational operators:\n";
  std::cout << "a1 " << (a1 < a2 ? "<" : a1>a2 ? ">" : "==") << " a2\n";
  std::cout << "a1 " << (a1 < a3 ? "<" : a1>a3 ? ">" : "==") << " a3\n";
  std::cout << "a1 " << (a1 != a2 ? "!=" : "==") << " a2\n";
  std::cout << "a1 " << (a1 != a3 ? "!=" : "==") << " a3\n";

  std::cout << "a2 " << (a2 < a1 ? "<" : a2>a1 ? ">" : "==") << " a1\n";
  std::cout << "a2 " << (a2 < a3 ? "<" : a2>a3 ? ">" : "==") << " a3\n";
  std::cout << "a2 " << (a2 != a1 ? "!=" : "==") << " a1\n";
  std::cout << "a2 " << (a2 != a3 ? "!=" : "==") << " a3\n";

  std::cout << "a3 " << (a3 < a1 ? "<" : a3>a1 ? ">" : "==") << " a1\n";
  std::cout << "a3 " << (a3 < a2 ? "<" : a3>a2 ? ">" : "==") << " a2\n";
  std::cout << "a3 " << (a3 != a1 ? "!=" : "==") << " a1\n";
  std::cout << "a3 " << (a3 != a2 ? "!=" : "==") << " a2\n";

  std::cout << "Testing non-member relational operators:\n";
  a1 = a2; a1[4] = 55;
  PRINT_ELEMENTS(a1, "a1: ", " ");
  PRINT_ELEMENTS(a2, "a2: ", " ");
  std::cout << "a1 " << (a1 < a2 ? "<" : a1>a2 ? ">" : "==") << " a2\n";
  std::cout << "a1 " << (a1 < a3 ? "<" : a1>a3 ? ">" : "==") << " a3\n";
}

void test_with_pair() {
  std::cout << "------------------TEST_WITH_PAIR-----------------------------\n";

  using PSS = std::pair<std::string, std::string>;
  using ARR_PSS4 = hlp2::Array<PSS,4>;

  // Test: default construction with initializer list
  ARR_PSS4 a;
  a[0] = std::make_pair("Seattle", "WA");   a[1] = std::make_pair("Portland", "OR");
  a[2] = std::make_pair("Vancouver", "BC"); a[3] = std::make_pair("Toronto", "ON");
  // Test: begin() and end()
  PRINT_ELEMENTS(a.begin(), a.end(), "a: ", " ");
  std::cout << "sizeof(a) == " << sizeof(a) << '\n';
  // Test: fill()
  a.fill(PSS("Montreal", "PQ"));
  // Test: cbegin() and cend()
  PRINT_ELEMENTS(a, "a: ", " ");

  // Test: fill()
  ARR_PSS4 b {
    std::make_pair("San Jose", "PR"), std::make_pair("San Antonio", "TX"),
    std::make_pair("San Luis Obispo", "CA"), std::make_pair("San Carlos", "AZ")
  };
  // Test: operator[]()
  PRINT_ELEMENTS<ARR_PSS4,4>(b, "b: ", " ");

  // Test: swap()
  a.swap(b);
  PRINT_ELEMENTS(a, "a: ", " ");
  PRINT_ELEMENTS(b, "b: ", " ");

  // Test: empty(), size(), front() and back()
  std::cout << std::boolalpha << b.empty() << '\n';
  if (!b.empty()) {
    b.front() = std::make_pair("Santa Barbara", "CA");
    b.back()  = std::make_pair("Santa Rosa", "CA");
    std::cout << "b: ";
    for (size_t i{}; i < b.size(); ++i) {
      std::cout << b[i] << " ";
    }
    std::cout << "\n";
  }

  // Test: default assignment
  ARR_PSS4 const cb = b;
  // Test: operator[]() const
  PRINT_ELEMENTS<ARR_PSS4,4>(cb, "cb: ", " ");

  // Test non-member functions:
  // hlp2::swap(), hlp2::op==(), hlp2::op!=(), hlp2::op<(), hlp2::op>()
  ARR_PSS4 a1{std::make_pair("Seattle","WA"), std::make_pair("Portland","OR"),
              std::make_pair("Vancouver","BC"), std::make_pair("Toronto","ON")};
  ARR_PSS4 a2{std::make_pair("San Jose", "PR"), std::make_pair("San Antonio", "TX"),
              std::make_pair("San Luis Obispo", "CA"), std::make_pair("San Carlos", "AZ")};
  ARR_PSS4 a3{a2};
  PRINT_ELEMENTS(a1, "a1: ", " ");
  PRINT_ELEMENTS(a2, "a2: ", " ");
  std::cout << "Testing non-member swap:\n";
  hlp2::swap(a1, a2);
  PRINT_ELEMENTS(a1, "a1: ", " ");
  PRINT_ELEMENTS(a2, "a2: ", " ");

  std::cout << "Testing non-member swap:\n";
  hlp2::swap(a1, a2);
  PRINT_ELEMENTS(a1, "a1: ", " ");
  PRINT_ELEMENTS(a2, "a2: ", " ");
  PRINT_ELEMENTS(a3, "a3: ", " ");

  std::cout << "Testing non-member relational operators:\n";
  std::cout << "a1 " << (a1 < a2 ? "<" : a1>a2 ? ">" : "==") << " a2\n";
  std::cout << "a1 " << (a1 < a3 ? "<" : a1>a3 ? ">" : "==") << " a3\n";
  std::cout << "a1 " << (a1 != a2 ? "!=" : "==") << " a2\n";
  std::cout << "a1 " << (a1 != a3 ? "!=" : "==") << " a3\n";

  std::cout << "a2 " << (a2 < a1 ? "<" : a2>a1 ? ">" : "==") << " a1\n";
  std::cout << "a2 " << (a2 < a3 ? "<" : a2>a3 ? ">" : "==") << " a3\n";
  std::cout << "a2 " << (a2 != a1 ? "!=" : "==") << " a1\n";
  std::cout << "a2 " << (a2 != a3 ? "!=" : "==") << " a3\n";

  std::cout << "a3 " << (a3 < a1 ? "<" : a3>a1 ? ">" : "==") << " a1\n";
  std::cout << "a3 " << (a3 < a2 ? "<" : a3>a2 ? ">" : "==") << " a2\n";
  std::cout << "a3 " << (a3 != a1 ? "!=" : "==") << " a1\n";
  std::cout << "a3 " << (a3 != a2 ? "!=" : "==") << " a2\n";

  std::cout << "Testing non-member relational operators:\n";
  a1 = a2; a1[3] = std::make_pair("Toronto", "ON");
  PRINT_ELEMENTS(a1, "a1: ", " ");
  PRINT_ELEMENTS(a2, "a2: ", " ");
  std::cout << "a1 " << (a1 < a2 ? "<" : a1>a2 ? ">" : "==") << " a2\n";
  std::cout << "a1 " << (a1 < a3 ? "<" : a1>a3 ? ">" : "==") << " a3\n";
}

// C-style function to return value obtained by subtracting 99 from parameter ...
int c_sub99(int ptr) {
  return ptr-99;
}

// C-style function that applies callback function to each array element ...
void c_xform(int *first, size_t N, int (*pf)(int)) {
  for (size_t i{}; i < N; ++i) {
    *first = pf(*first);
    ++first;
  }
}

// C-style function to return sum of elements of an int array
int c_sum(int const *first, size_t N) {
  int total{};
  for (size_t i{}; i < N; ++i) {
    total += *first++;
  }
  return total;
}

template <typename T1, typename T2>
std::ostream& operator<<(std::ostream& os, std::pair<T1, T2>const& rhs) {
  os << "<" << rhs.first << ", " << rhs.second << ">";
  return os;
}

template <typename It>
void PRINT_ELEMENTS(It first, It last, std::string const& prefix,
                    std::string const& sep, std::string const& postfix) {
  std::cout << prefix;
  while (first != last) {
    std::cout << *first << sep;
    ++first;
  }
  std::cout << postfix;
}

template <typename Cont>
void PRINT_ELEMENTS(Cont const& cont, std::string const& prefix, 
                    std::string const& sep, std::string const& postfix) {
  std::cout << prefix;
  typename Cont::const_iterator first = cont.cbegin(), last = cont.cend();
  while (first != last) {
    std::cout << *first << sep;
    ++first;
  }
  std::cout << postfix;
}

template <typename Cont, size_t N>
void PRINT_ELEMENTS(Cont const& cont, std::string const& prefix,
                  std::string const& sep, std::string const& postfix) {
  std::cout << prefix;
  for (typename Cont::size_type i{}; i < N; ++i) {
    std::cout << cont[i] << sep;
  }
  std::cout << postfix;
}

} // end anonymous namespace
