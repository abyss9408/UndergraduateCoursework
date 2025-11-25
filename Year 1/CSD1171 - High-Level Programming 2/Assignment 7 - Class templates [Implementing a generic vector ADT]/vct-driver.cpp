#include <iostream> // std::cout, std::endl
#include <string>   // std::string
#include <array>    // std::array
#include <utility>  // std::pair
#include <cctype>   // std::toupper
#include "vct.hpp"  // hlp2::vector<T>
#include "str.hpp"  // hlp2::Str

namespace { // anonymous namespace
  // write to output stream a pair<T1,T2>
  template <typename T1, typename T2>
  std::ostream& operator<<(std::ostream& os, std::pair<T1, T2>const& rhs);
  // write to output stream a hlp2::vector<T>
  template <typename T>
  std::ostream& operator<<(std::ostream& os, hlp2::vector<T> const& rhs);
  // write to output stream a hlp2::vector<T>
  template <typename T>
  void Print(hlp2::vector<T> const& lhs,
             std::string const& newline = "\n", std::ostream& os=std::cout);
  // swap values in two variables
  template <typename T>
  void swap(T& lhs, T& rhs);

  void TestPushPop();    // correct output is in test-pushpop.txt
  void TestSubscripts(); // correct output is in test-subscript.txt
  void TestCopyAssign(); // correct output is in test-copyassign.txt
  void TestResize();     // correct output is in test-resize.txt
} // end anonymous namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << "Usage: ./vector.out test-number-between-0-and-4\n";
    return 0;
  }

  int const num_of_tests {4};
  int test_num {0};
  if (argc > 1) {
    test_num = std::stoi(argv[1]);
    test_num = (test_num < 0) ? 0 : test_num;
    test_num = (test_num > num_of_tests) ? num_of_tests : test_num;
  }

  using TPF = void (*)();
  std::array<TPF, num_of_tests> Tests {
    TestPushPop,     // 1 - correct output is in test-pushpop.txt
    TestSubscripts,  // 2 - correct output is in test-subscript.txt
    TestCopyAssign,  // 3 - correct output is in test-copyassign.txt
    TestResize       // 4 - correct output is in test-resize.txt
  };

  if (test_num == 0) { // correct output when argv[1] is "0" is in test-all.txt
    for (TPF x : Tests) {
      x();
    }
  } else {
    Tests[test_num - 1]();
  }
}

namespace { // anonymous namespace ...

void TestPushPop() {
  std::cout << "********** TestPushPop **********\n";

  // dealing with ints
  std::cout << "push/pop int\n";
  hlp2::vector<int> a;
  std::cout << "a: " << a;
  std::cout << "push_back 5 ints\n";
  for (int i : {-1, 1, -2, 2, -3}) { a.push_back(i); Print(a); }
  std::cout << "now push_back 5 more ints:\n";
  for (int i : {-10, 10, -20, 20, 30}) { a.push_back(-i); Print(a); }
  std::cout << "now pop_back ints until empty:\n";
  while (!a.empty()) { a.pop_back(); Print(a); }
  std::cout << a << '\n';

  // deal with chars
  std::cout << "push/pop char\n";
  hlp2::vector<char> b{'s','i','n','g','a'};
  b.push_back('p'); b.push_back('o'); b.push_back('r'); b.push_back('e');
  std::cout << b << "pop_back chars until empty:\n";
  while (!b.empty()) { b.pop_back(); std::cout << b; }
  std::cout << b << "\n";

  // deal with hlp2::Str
  std::cout << "push/pop hlp2::Str\n";
  hlp2::vector<hlp2::Str> c{"Namaste", "Hello", "Bonjour", "Hola", "Annyeonghaseyo"};
  c.push_back("Nin Hao"); c.push_back("Konnichiwa"); c.push_back("Guten Tag");
  std::cout << c << "pop_back strings until empty:\n";
  while (!c.empty()) { c.pop_back(); std::cout << c; }
  Print(c); std::cout << "\n";

  // deal with doubles
  std::cout << "push/pop double\n";
  hlp2::vector<double> d{2.1, -3.2, 4.3, -5.4, 5.5, -6.6, 7.7, -8.8};
  d.push_back(9.9); d.push_back(10.0);
  std::cout << d << "pop_back doubles until empty:\n";
  while (!d.empty()) { d.pop_back(); std::cout << d; }
  std::cout << d << "\n";

  std::cout << "push/pop pair<hlp2::Str, std::string>\n";
  hlp2::vector<std::pair<hlp2::Str, std::string>> e {
    std::make_pair("Seattle", "WA"), std::make_pair("Portland", "OR"),
    std::make_pair("Vancouver", "BC"), std::make_pair("Santa Barbara", "CA")
  };
  e.push_back(std::make_pair("Toronto", "ON"));   e.push_back(std::make_pair("Montreal", "PQ"));
  std::cout << e << "pop_back pair<hlp2::Str, std::string> until empty:\n";
  while (!e.empty()) { e.pop_back(); Print(e); }
  std::cout << e;
}

void TestSubscripts() {
  std::cout << "\n********** TestSubscripts **********\n";

  hlp2::vector<int> a{0, 2, 4, 6, 8, 10, 12, 14, 16, 18};
  std::cout << "a     : " << a;
  for (hlp2::vector<int>::size_type i{0}; i < a.size(); ++i) { a[i] = a[i] * 3; }
  std::cout << "a[i]*2: " << a << '\n';

  hlp2::vector<char> b{'<','h','+','e','?','l','&','l','*','o','>'};
  std::cout << "b[lower case]: "; Print(b);
  for (hlp2::vector<char>::size_type i{0}; i < b.size(); ++i) {
    b[i] = std::toupper(b[i]);
  }
  std::cout << "b[upper case]: " << b << "\n";

  hlp2::vector<hlp2::Str> c{"Hello", "Bonjour", "Hola", "Nin hao", "Konnichiwa", "Guten Tag"};
  std::cout << "c[lower case]: " << c;
  for (hlp2::vector<std::string>::size_type i = 0; i < c.size(); ++i) {
    for (hlp2::Str::reference ch : c[i]) { ch = std::toupper(ch); }
  }
  std::cout << "c[upper case]: " << c << "\n";

  hlp2::vector<double> d{1., 2., 3., 4., 5.};
  std::cout << "d   : " << d;
  for (hlp2::vector<double>::size_type i{0}; i < d.size(); ++i) {
    d[i] = d[i] * 3.14159;
  }
  std::cout << "d*pi: " << d << '\n';

  hlp2::vector<std::pair<std::string, hlp2::Str>> e{
    std::make_pair("Seattle", "WA"), std::make_pair("Portland", "OR"), std::make_pair("Vancouver", "BC"),
    std::make_pair("Santa Barbara", "CA"), std::make_pair("Toronto", "ON"), std::make_pair("Montreal", "PQ")
  };
  std::cout << "e[AB]: " << e;
  for (hlp2::vector<std::pair<std::string, hlp2::Str>>::size_type i{0}; i < e.size(); ++i) {
    for (char& x : e[i].second) { x = std::tolower(x); }
  }
  std::cout << "e[ab]: " << e;
}

void TestCopyAssign() {
  std::cout << "\n********** TestCopyAssign **********\n";
  
  {
    // call hlp2::vector<T>::swap()
    hlp2::vector<int> a1{0, 2, 4, 6, 8, 10, 12, 14, 16, 18}, a2;
    a2.push_back(1); a2.push_back(3); a2.push_back(5); a2.push_back(7);
    std::cout << "before vector<T>::swap\n" << "a1: " << a1 << "a2: " << a2;
    a2.swap(a1);
    std::cout << "after vector<T>::swap\n" << "a1: " << a1 << "a2: " << a2 << '\n';
  }

  {
    // call ::swap<T> [which uses hlp2::vector<T>::op=()]
    hlp2::vector<int> a1{0, 2, 4, 6, 8, 10, 12, 14, 16, 18}, a2{1, 3, 5, 7};
    std::cout << "before ::swap\n" << "a1: " << a1 << "a2: " << a2;
    ::swap(a1, a2);
    std::cout << "after ::swap\n"  << "a1: " << a1 << "a2: " << a2 << '\n';
  }

  {
    // call hlp2::vector<T>::swap()
    hlp2::vector<char> b1{'<','~','@','!','+','e','?','#','&','%'}, b2;
    b2.push_back('*'); b2.push_back('^'); b2.push_back('('); b2.push_back(')'); b2.push_back('>');
    std::cout << "before hlp2::vector<T>::swap\n" << "b1: " << b1 << "b2: " << b2;
    b2.swap(b1);
    std::cout << "after hlp2::vector<T>::swap\n"  << "b1: " << b1 << "b2: " << b2 << '\n';
  }

  {
    // call ::swap<T> [which uses hlp2::vector<T>::op=()]
    hlp2::vector<char> b1{'<','~','@','!','+','e','?','#','&','%','*','^','(',')','>'}, b2{};
    std::cout << "before ::swap\n" << "b1: " << b1 << "b2: " << b2;
    //::swap(b1, b2);
    ::swap(b1, b2);
    std::cout << "after ::swap\n"  << "b1: " << b1 << "b2: " << b2 << '\n';
  }

  {
    // call hlp2::vector<T>::swap()
    hlp2::vector<std::string> c1{"Hello", "Bonjour", "Hola", "Nin hao", "Konnichiwa"}, c2{};
    c2.push_back("Guten Tag"); c2.push_back("Namaste");
    std::cout << "before hlp2::vector<T>::swap\n" << "c1: " << c1 << "c2: " << c2;
    c2.swap(c1);
    std::cout << "after hlp2::vector<T>::swap\n"  << "c1: " << c1 << "c2: " << c2 << '\n';
  }

  {
    // call ::swap<T> [which uses hlp2::vector<T>::op=()]
    hlp2::vector<std::string> c1{"Hello", "Bonjour", "Hola", "Nin hao", "Konnichiwa"}, c2{};
    c2.push_back("Guten Tag"); c2.push_back("Namaste");
    std::cout << "before ::swap\n" << "c1: " << c1 << "c2: " << c2;
    ::swap(c1, c2);
    std::cout << "after ::swap\n"  << "c1: " << c1 << "c2: " << c2 << '\n';
  }

  {
    // call hlp2::vector<T>::swap()
    hlp2::vector<double> d1{2.05, 4.05, 6.05, 8.05, 10.05, 12.05}, d2{-12.1,-12.2,12.3};
    d2.push_back(-12.4); d2.push_back(12.5); d2.push_back(-12.6);
    std::cout << "before hlp2::vector<T>::swap\n" << "d1: " << d1 << "d2: " << d2;
    d2.swap(d1);
    std::cout << "after hlp2::vector<T>::swap\n"  << "d1: " << d1 << "d2: " << d2 << '\n';
  }

  {
    // call ::swap<T> [which uses hlp2::vector<T>::op=()]
    hlp2::vector<double> d1{2.05, 4.05, 6.05, 8.05, 10.05, 12.05}, d2{-12.1,-12.2,12.3};
    d2.push_back(-12.4); d2.push_back(12.5); d2.push_back(-12.6);
    std::cout << "before ::swap\n" << "d1: " << d1 << "d2: " << d2;
    ::swap(d1, d2);
    std::cout << "after ::swap\n"  << "d1: " << d1 << "d2: " << d2 << '\n';
  }

  {
    using PSS = std::pair<std::string, hlp2::Str>;
    hlp2::vector<PSS> e1{std::make_pair("Seattle", "WA"), std::make_pair("Portland", "OR")}, e2;
    e2.push_back(std::make_pair("Vancouver", "BC")); e2.push_back(std::make_pair("Santa Barbara", "CA"));
    e2.push_back(std::make_pair("Toronto", "ON"));   e2.push_back(std::make_pair("Montreal", "PQ"));
    std::cout << "before hlp2::vector<T>::swap\n" << "e1: " << e1 << "e2: " << e2;
    e2.swap(e1);
    std::cout << "after hlp2::vector<T>::swap\n"  << "e1: " << e1 << "e2: " << e2 << "\n";
  }

  {
    using PSS = std::pair<std::string, hlp2::Str>;
    hlp2::vector<PSS> e1{std::make_pair("Seattle", "WA"), std::make_pair("Portland", "OR")}, e2;
    e2.push_back(std::make_pair("Vancouver", "BC")); e2.push_back(std::make_pair("Santa Barbara", "CA"));
    e2.push_back(std::make_pair("Toronto", "ON"));   e2.push_back(std::make_pair("Montreal", "PQ"));
    std::cout << "before ::swap\n" << "e1: " << e1 << "e2: " << e2;
    ::swap(e1, e2);
    std::cout << "after ::swap\n"  << "e1: " << e1 << "e2: " << e2;
  }
}

void TestResize() {
  std::cout << "\n********** TestResize **********\n";

  hlp2::vector<int> a{0, 2, 4, 6, 8, 10, 12, 14, 16, 18};
  std::cout << "a before resize to 0: " << a;
  a.resize(0);
  std::cout << "a after  resize to 0: " << a << '\n';
  
  hlp2::vector<char> b{'<','~','@','!','+','e','?','#','&','%','*','^','(',')','>'};
  std::cout << "b before resize to 1: " << b;
  b.resize(1);
  std::cout << "b after  resize to 1: " << b << "\n";

  hlp2::vector<std::string> c{"Hello", "Bonjour", "Hola", "Nin hao", "Konnichiwa", "Guten Tag"};
  std::cout << "c before resize to 2: " << c;
  c.resize(2);
  std::cout << "c after  resize to 2: " << c << "\n";

  hlp2::vector<double> d{2.05, 4., 6., 6., 8., 10., 6., 12., -6., 14., 16., 6., 6.};
  std::cout << "d before resize to 3: " << d;
  d.resize(3);
  std::cout << "d after  resize to 3: " << d << "\n";

  using PSS = std::pair<std::string, hlp2::Str>;
  hlp2::vector<PSS> e{
    std::make_pair("Seattle", "WA"),   std::make_pair("Portland", "OR"),
    std::make_pair("Vancouver", "BC"), std::make_pair("Santa Barbara", "CA"),
    std::make_pair("Toronto", "ON"),   std::make_pair("Montreal", "PQ")
  };
  std::cout << "e before resize to 4: " << e;
  e.resize(4);
  std::cout << "e after  resize to 4: " << e << "\n";

  hlp2::vector<std::string> f{"Seattle", "Singapore"};
  std::cout << "f before resize to 5: " << f;
  f.resize(5);
  std::cout << "f after resize to 5: " << f;

  // elements with subscripts 2, 3, and 4 are default constructed ...
  f[2] = "Sydney"; f[3] = "San Jose"; f[4] = "San Antonio";
  // push_back should increase size to 6
  f.push_back("Santiago");
  std::cout << "f after push: " << f << "\n";
}

template <typename T1, typename T2>
std::ostream& operator<<(std::ostream& os, std::pair<T1, T2>const& rhs) {
  os << "<" << rhs.first << ", " << rhs.second << ">";
  return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, hlp2::vector<T> const& rhs) {
  for (typename hlp2::vector<T>::const_reference x : rhs) {
    os << x << " ";
  }
  os << "[size=" << rhs.size() << ", capacity="
     << rhs.capacity() << ", allocs=" << rhs.allocations() << "]" << '\n';
  return os;
}

template <typename T>
void Print(hlp2::vector<T> const& lhs, std::string const& newline, std::ostream& os) {
  for (typename hlp2::vector<T>::size_type i{}; i < lhs.size(); ++i) {
    //std::cout << lhs[i] << " ";
    os << lhs[i] << " ";
  }
  //std::cout << "[size=" << lhs.size() << ", capacity=" <<  lhs.capacity() << ", allocs="
  os << "[size=" << lhs.size() << ", capacity=" <<  lhs.capacity() << ", allocs="
                        << lhs.allocations() << "]" << newline;
}

template <typename T>
void swap(T& lhs, T& rhs) {
  T tmp{lhs};
  lhs = rhs;
  rhs = tmp;
}

} // end anonymous namespace
