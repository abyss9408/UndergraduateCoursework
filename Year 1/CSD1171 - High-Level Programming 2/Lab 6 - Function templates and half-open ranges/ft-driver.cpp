#include <iostream> // ostream, cout, endl
#include <array>
#include <string>
#include <forward_list>
#include <list>
#include "student.hpp"
#include "ft.hpp"

namespace {
	void TestSwap();
	void TestSwapRanges();
	void TestRemove();
	void TestCount();
	void TestFind();
	void TestCopy();
	void TestFill();
	void TestReplace();
	void TestMin(); 
	void TestMax(); 
	void TestEqual();
	void TestSum();
} // end anonymous namspace

int main() {
  TestSwap();
  TestSwapRanges();
  TestRemove();
  TestCount();
  TestFind();
  TestCopy();
  TestFill();
  TestReplace();
  TestMin();
  TestMax();
  TestEqual();
  TestSum();
}

namespace {
void TestSwap() {
  std::cout << "***** TestSwap1 *****" << "\n";
  int a = 5, b = 8;
  std::cout << "a = " << a << ", b = " << b << "\n";
  hlp2::swap(a, b);
  std::cout << "a = " << a << ", b = " << b << "\n";

  std::cout << "***** TestSwap2 *****" << "\n";
  Student s1 = {"jdoe", 20, 3, 3.82}; std::cout << "s1: " << s1;
  Student s2 = {"fbar", 22, 1, 1.28}; std::cout << "s2: " << s2;
  hlp2::swap(s1, s2);
  std::cout << "s1: " << s1; std::cout << "s2: " << s2;
}

void TestSwapRanges() {
  std::cout << "***** SwapRanges *****" << "\n";
  std::array<int, 11> i1 {-1, 2, 6, -1, 9, 5, 7, -1, -1, 8, -1};
  std::array<int, 11> i2 {3, 6, 8, 0, -1, 5, -3, -9, 3, 5, 8};
  hlp2::display(i1.data(), i1.data()+i1.size());
  hlp2::display(i2.data(), i2.data()+i2.size());
  hlp2::swap_ranges(i1.data(), i1.data()+i1.size(), i2.data());
  hlp2::display(i1.data(), i1.data()+i1.size());
  hlp2::display(i2.data(), i2.data()+i2.size());

  std::array<std::string, 6>  as1 {"a", "b", "c", "d", "e", "f"};
  std::array<std::string, 10> as2 {"g", "h", "i", "j", "k", "l", "m", "n", "o", "p"};
  hlp2::display(as1.data(), as1.data()+as1.size());
  hlp2::display(as2.data(), as2.data()+as2.size());
  hlp2::swap_ranges(as1.data(), as1.data()+3, as2.data());
  hlp2::swap_ranges(as1.data()+3, as1.data()+as1.size(), as2.data()+3);
  hlp2::display(as1.data(), as1.data()+as1.size());
  hlp2::display(as2.data(), as2.data()+as2.size());

  std::array<Student, 5> astu1 {
                   Student{"jdoe", 20, 3, 3.82},
                   Student{"fbar", 22, 1, 1.28},
                   Student{"wxyz", 19, 3, 1.59},
                   Student{"abcd", 20, 1, 1.99},
                   Student{"jbar", 22, 2, 3.38}
                 };
  std::array<Student, 5> astu2 {
                   Student{"eodj", 21, 3, 2.38},
                   Student{"rabf", 20, 2, 2.81},
                   Student{"zyxw", 19, 2, 2.95},
                   Student{"dcba", 17, 1, 2.91},
                   Student{"rabj", 22, 3, 2.5}
                 };
  print(astu1.data(), astu1.data()+astu1.size()); std::cout << "\n";
  print(astu2.data(), astu2.data()+astu2.size()); std::cout << "\n\n";
  hlp2::swap_ranges(astu1.data(), astu1.data()+3, astu2.data());
  hlp2::swap_ranges(astu1.data()+3, astu1.data()+astu1.size(), astu2.data()+3);
  print(astu1.data(), astu1.data()+astu1.size()); std::cout << "\n";
  print(astu2.data(), astu2.data()+astu2.size()); std::cout << "\n\n";
}

void TestRemove() {
  std::cout << "***** Remove1 *****" << "\n";
  std::array<int, 7> i1 {5, -7, 4, 10, -21, 15, 9};
  hlp2::display(i1.data(), i1.data()+i1.size());
  int *newend = hlp2::remove(i1.data(), i1.data()+i1.size(),  -1);
  std::cout << "remove " << -1 << ", new list: ";
  hlp2::display(i1.data(), newend);

  std::cout << "***** Remove2 *****" << "\n";
  i1 = {5, -7, 4, 10, -7, 15, 9};
  hlp2::display(i1.data(), i1.data()+i1.size());
  newend = hlp2::remove(i1.data(), i1.data()+i1.size(),  -7);
  std::cout << "remove " << -7 << ", new list: ";
  hlp2::display(i1.data(), newend);

  std::cout << "***** Remove3 *****" << "\n";
  i1 = {-1, -1, -1, -1, -1, -1, -1};
  hlp2::display(i1.data(), i1.data()+i1.size());
  newend = hlp2::remove(i1.data(), i1.data()+i1.size(),  -1);
  std::cout << "remove " << -1 << ", new list: ";
  hlp2::display(i1.data(), newend);

  std::cout << "***** Remove4 *****" << "\n";
  std::array<char, 25> as1 {'l','e','a','r','n','i','n','g','a','b','o','u','t','c','+','+','t','e','m','p','l','a','t','e','s'};
  hlp2::display(as1.data(), as1.data()+as1.size());
  char *ch_newend = hlp2::remove(as1.data(), as1.data()+as1.size(), '+');
  hlp2::display(as1.data(), ch_newend);
  ch_newend = hlp2::remove(as1.data(), ch_newend, 'e');
  hlp2::display(as1.data(), ch_newend);
  ch_newend = hlp2::remove(as1.data(), ch_newend, 't');
  hlp2::display(as1.data(), ch_newend);
  ch_newend = hlp2::remove(as1.data(), ch_newend, 'a');
  hlp2::display(as1.data(), ch_newend);
  ch_newend = hlp2::remove(as1.data(), ch_newend, 's');
  hlp2::display(as1.data(), ch_newend);
  ch_newend = hlp2::remove(as1.data(), ch_newend, 'l');
  hlp2::display(as1.data(), ch_newend);
}

void TestCount() {
  std::cout << "***** Count1 *****" << "\n";
  std::array<int, 11> ai {-1, 2, 6, -1, 9, 5, 7, -1, -1, 8, -1};
  hlp2::display(ai.data(), ai.data()+ai.size());
  int c = hlp2::count(ai.data(), ai.data()+ai.size(), -1);
  std::cout << "Count of " << -1 << " is " << c << "\n";

  std::cout << "***** Count2 *****" << "\n";
  ai = {-1, 2, 6, -1, 9, 5, 7, -1, -1, 8, -1};
  hlp2::display(ai.data(), ai.data()+ai.size());
  c = hlp2::count(ai.data(), ai.data()+ai.size(), 50);
  std::cout << "Count of " << 50 << " is " << c << "\n";

  std::cout << "***** Count3 *****" << "\n";
  std::array<char, 25> ac {'l','e','a','r','n','i','n','g','a','b','o','u',
                       't','c','+','+','t','e','m','p','l','a','t','e','s'};
  hlp2::display(ac.data(), ac.data()+ac.size());
  c = hlp2::count(ac.data(), ac.data()+ac.size(), '+');
  std::cout << "Count of " << '+' << " is " << c << "\n";
  c = hlp2::count(ac.data()+5, ac.data()+ac.size(), 'n');
  std::cout << "Count of " << 'n' << " is " << c << "\n";

  std::cout << "***** Count4 *****" << "\n";
  std::array<std::string, 25> as {"l","e","a","r","n","i","n","g","a","b","o","u","t",
                              "c","+","+","t","e","m","p","l","a","t","e","s"};
  hlp2::display(as.data(), as.data()+as.size());
  c = hlp2::count(as.data(), as.data()+as.size(), "l");
  std::cout << "Count of " << "l" << " is " << c << "\n";
  c = hlp2::count(as.data()+2, as.data()+as.size(), "e");
  std::cout << "Count of " << "e" << " is " << c << "\n";
}

void TestFind() {
  std::cout << "***** Find1 *****" << "\n";
  const std::array<int, 11> cai = {-1, 2, 6, -1, 9, 5, 7, -1, -1, 8, -1};
  hlp2::display(cai.data(), cai.data()+cai.size());
  const int item = 9;
  const int *cpos = hlp2::find(cai.data(), cai.data()+cai.size(), item);
  if (cpos != cai.data()+cai.size()) {
    std::cout << "Item " << item << " is at index " << cpos-cai.data() << "\n";
  } else {
    std::cout << "Item " << item << " was not found" << "\n";
  }

  std::cout << "***** Find2 *****" << "\n";
  std::array<int, 11> ai = {-1, 2, 6, -1, 9, 5, 7, -1, -1, 8, -1};
  hlp2::display(ai.data(), ai.data() + ai.size());
  int *pos = hlp2::find(ai.data(), ai.data() + ai.size(), item);
  if (pos != ai.data() + ai.size()) {
    std::cout << "Changing " << item << " to 100" << "\n";
    *pos = 100;
  } else {
    std::cout << "Item " << item << " was not found" << "\n";
  }
  hlp2::display(ai.data(), ai.data() + ai.size());
  
  std::cout << "***** Find3 *****" << "\n";
  hlp2::display(cai.data(), cai.data()+cai.size());
  const int item2 = 19;
  cpos = hlp2::find(cai.data(), cai.data()+cai.size(), item2);
  if (cpos != cai.data()+cai.size()) {
    std::cout << "Item " << item2 << " is " << *pos << "\n";
  } else {
    std::cout << "Item " << item2 << " was not found" << "\n";
  }

  std::cout << "***** Find4 *****" << "\n";
  std::array<char, 25> ac {'l','e','a','r','n','i','n','g','a','b','o','u',
                       't','c','+','+','t','e','m','p','l','a','t','e','s'};
  hlp2::display(ac.data(), ac.data()+ac.size());
  const char item3 = 'n';
  char *pcpos = hlp2::find(ac.data(), ac.data()+ac.size(), item3);
  if (pcpos != ac.data()+ac.size()) {
    std::cout << "Item " << item3 << " is at index " << pcpos-ac.data() << "\n";
    std::cout << "Changing " << item3 << " to Z" << "\n";
    *pcpos = 'Z';
  } else {
    std::cout << "Item " << item3 << " was not found" << "\n";
  }
  hlp2::display(ac.data(), ac.data()+ac.size());

  std::cout << "***** Find5 *****" << "\n";
  std::array<std::string, 25> as {"l","e","a","r","n","i","n","g","a","b","o","u","t",
                              "c","+","+","t","e","m","p","l","a","t","e","s"};
  hlp2::display(as.data(), as.data()+as.size());
  const std::string item4 {"m"};
  std::string *pspos = hlp2::find(as.data(), as.data()+as.size(), item4);
  if (pspos != as.data()+as.size()) {
    std::cout << "Item " << item4 << " is at index " << pspos-as.data() << "\n";
    std::cout << "Changing " << item4 << " to Montreal" << "\n";
    *pspos = "Montreal";
  } else {
    std::cout << "Item " << item4 << " was not found" << "\n";
  }
  hlp2::display(as.data(), as.data()+as.size());
}

void TestCopy() {
  std::cout << "***** Copy1 *****" << "\n";
  const std::array<short, 11> ash {-1, 2, 6, -1, 9, 5, 7, -1, -1, 8, -1};
  std::array<short,20> ash2 {0};
  hlp2::display(ash.data(), ash.data()+ash.size());
  hlp2::display(ash2.data(), hlp2::copy(ash.data(), ash.data()+ash.size(), ash2.data()));

  std::cout << "***** Copy2 *****" << "\n";
  std::array<char, 25> ac {'l','e','a','r','n','i','n','g','a','b','o','u',
                       't','c','+','+','t','e','m','p','l','a','t','e','s'};
  hlp2::display(ac.data(), ac.data()+ac.size());
  std::array<char, 10> ac2 {'z','z','z','z','z','z','z','z','z','z'};
  hlp2::display(ac2.data(), hlp2::copy(ac.data()+10, ac.data()+20, ac2.data()));
  
  std::cout << "***** Copy3 *****" << "\n";
  std::array<std::string, 25> as {"l","e","a","r","n","i","n","g","a","b","o","u","t",
                              "c","+","+","t","e","m","p","l","a","t","e","s"};
  hlp2::display(as.data(), as.data()+as.size());
  std::array<std::string, 25> as2{};
  hlp2::display(as2.data(), hlp2::copy(as.data(), as.data(), as2.data()));
}

void TestFill() {
  std::cout << "***** Fill1 *****" << "\n";
  std::array<int, 10> ai{0};
  hlp2::fill(ai.data(), ai.data()+ai.size(), 12);
  hlp2::display(ai.data(), ai.data()+ai.size());

  std::cout << "***** Fill2 *****" << "\n";
  std::array<char, 30> ac{};
  hlp2::fill(ac.data(), ac.data()+ac.size(), '*');
  hlp2::display(ac.data(), ac.data()+ac.size());

  std::cout << "***** Fill3 *****" << "\n";
  std::array<std::string, 30> as{};
  hlp2::fill(as.data(), as.data()+as.size(), "*");
  hlp2::display(as.data(), as.data()+as.size());
}

void TestReplace() {
  std::cout << "***** Replace1 *****" << "\n";
  std::array<int, 11> ai {-1, 2, 6, -1, 9, 5, 7, -1, -1, 8, -1};
  hlp2::display(ai.data(), ai.data()+ai.size());
  int olditem {-1}, newitem {-8};
  std::cout << "Replacing " << olditem << " with " << newitem << "\n";
  hlp2::replace(ai.data(), ai.data()+ai.size(), olditem, newitem);
  hlp2::display(ai.data(), ai.data()+ai.size());

  std::cout << "***** Replace2 *****" << "\n";
  std::array<char, 25> ac {'l','e','a','r','n','i','n','g','a','b','o','u',
                       't','c','+','+','t','e','m','p','l','a','t','e','s'};
  hlp2::display(ac.data(), ac.data()+ac.size());
  std::cout << "Replacing " << 'p' << " with " << 'P' << "\n";
  hlp2::replace(ac.data(), ac.data()+ac.size(), 'p', 'P');
  hlp2::display(ac.data(), ac.data()+ac.size());
  
  std::cout << "***** Replace3 *****" << "\n";
  std::array<std::string, 25> as {"l","e","a","r","n","i","n","g","a","b","o","u","t",
                              "c","+","+","t","e","m","p","l","a","t","e","s"};
  hlp2::display(as.data(), as.data()+as.size());
  std::cout << "Replacing " << "p" << " with " << "P" << "\n";
  hlp2::replace(as.data(), as.data()+as.size(), "p", "P");
  hlp2::display(as.data(), as.data()+as.size());
}

void TestMin() {
  std::cout << "***** Min1 *****" << "\n";
  std::array<int, 11> ai {-1, 2, 6, -31, 9, 5, 7, -21, -1, 8, -10};
  hlp2::display(ai.data(), ai.data()+ai.size());
  int const *pos = hlp2::min_element(ai.data(), ai.data()+ai.size());
  std::cout << "The min element is: " << *pos << "\n";

  std::cout << "***** Min2 *****" << "\n";
  std::array<double, 11> ad {-1.1, 2.2, 6.3, -31.4, 9.5, 5.6, 7.7, -21.8, -1.9, 8.1, -10.2};
  hlp2::display(ad.data(), ad.data()+ad.size());
  double const *dpos = hlp2::min_element(ad.data(), ad.data()+ad.size());
  std::cout << "The min element is: " << *dpos << "\n";

  std::cout << "***** Min3 *****" << "\n";
  std::array<char, 25> ac {'l','e','a','r','n','i','n','g','a','b','o','u',
                       't','c','+','+','t','e','m','p','l','a','t','e','s'};
  hlp2::display(ac.data(), ac.data()+ac.size());
  char const *cpos = hlp2::min_element(ac.data(), ac.data()+ac.size());
  std::cout << "The min element is: " << *cpos << "\n";
  
  std::cout << "***** Min4 *****" << "\n";
  std::array<std::string, 25> as {"l","e","a","r","n","i","n","g","a","b","o","u","t",
                              "c","+","+","t","e","m","p","l","a","t","e","s"};
  hlp2::display(as.data(), as.data()+as.size());
  std::string const *spos = hlp2::min_element(as.data(), as.data()+as.size());
  std::cout << "The min element is: " << *spos << "\n";

  std::cout << "***** Min5 *****" << "\n";
  std::array<Student, 5> ast {
                  Student{"jdoe", 20, 3, 3.82},
                  Student {"fbar", 22, 1, 1.28},
                  Student{"wxyz", 19, 3, 1.59},
                  Student{"abcd", 20, 1, 1.99},
                  Student {"jbar", 22, 2, 3.38}
                 };

  print(ast.data(), ast.data()+ast.size());
  Student const *stpos = hlp2::min_element(ast.data(), ast.data()+ast.size());
  std::cout << "The min element is: " << *stpos;
}

void TestMax() {
  std::cout << "***** Max1 *****" << "\n";
  std::array<int, 11> ai {-1, 2, 6, -31, 9, 5, 7, -21, -1, 8, -10};
  hlp2::display(ai.data(), ai.data()+ai.size());
  int const *pos = hlp2::max_element(ai.data(), ai.data()+ai.size());
  std::cout << "The max element is: " << *pos << "\n";

  std::cout << "***** Max2 *****" << "\n";
  std::array<double, 11> ad {-1.1, 2.2, 6.3, -31.4, 9.5, 5.6, 7.7, -21.8, -1.9, 8.1, -10.2};
  hlp2::display(ad.data(), ad.data()+ad.size());
  double const *dpos = hlp2::max_element(ad.data(), ad.data()+ad.size());
  std::cout << "The max element is: " << *dpos << "\n";

  std::cout << "***** Max3 *****" << "\n";
  std::array<char, 25> ac {'l','e','a','r','n','i','n','g','a','b','o','u',
                       't','c','+','+','t','e','m','p','l','a','t','e','s'};
  hlp2::display(ac.data(), ac.data()+ac.size());
  char const *cpos = hlp2::max_element(ac.data(), ac.data()+ac.size());
  std::cout << "The max element is: " << *cpos << "\n";
  
  std::cout << "***** Max4 *****" << "\n";
  std::array<std::string, 25> as {"l","e","a","r","n","i","n","g","a","b","o","u","t",
                              "c","+","+","t","e","m","p","l","a","t","e","s"};
  hlp2::display(as.data(), as.data()+as.size());
  std::string const *spos = hlp2::max_element(as.data(), as.data()+as.size());
  std::cout << "The max element is: " << *spos << "\n";

  std::cout << "***** Max5 *****" << "\n";
  std::array<Student, 5> ast {
                   Student{"jdoe", 20, 3, 3.82},
                   Student{"fbar", 22, 1, 1.28},
                   Student{"wxyz", 19, 3, 1.59},
                   Student{"abcd", 20, 1, 1.99},
                   Student{"jbar", 22, 2, 3.38}
                 };
  print(ast.data(), ast.data()+ast.size());
  Student *stpos = hlp2::max_element(ast.data(), ast.data()+ast.size());
  std::cout << "The max element is: " << *stpos;
}

void TestEqual() {
  std::cout << "***** Equal1 *****" << "\n";
  std::array<int, 11> ai {-1, 2, 6, -1, 9, 5, 7, -1, -1, 8, -1};
  std::array<short, 11> ash {-1, 2, 6, -1, 9, 5, 7, -1, -1, 8, -1};
  
  hlp2::display(ai.data(), ai.data()+ai.size());
  hlp2::display(ash.data(), ash.data()+ash.size());
  bool same = hlp2::equal(ai.data(), ai.data()+ai.size(), ash.data());
  std::cout << "Arrays are " << (same ? "equal\n" : "not equal\n");

  std::cout << "***** Equal2 *****" << "\n";
  std::array<int, 7> ai2 {-1, 2, 6, -1, 9, 5, 7};
  std::array<short, 11> ash2 {-1, 2, 6, -1, 9, 5, 7, -1, -1, 8, -1};
  hlp2::display(ai2.data(), ai2.data()+ai2.size());
  hlp2::display(ash2.data(), ash2.data()+ash2.size());
  same = hlp2::equal(ai2.data(), ai2.data()+ai2.size(), ash2.data());
  std::cout << "Arrays are " << (same ? "equal\n" : "not equal\n");

  std::cout << "***** Equal3 *****" << "\n";
  std::array<int, 7> ai3 {-1, 2, 6, -1, 9, 5, 7};
  std::array<short, 7> ash3 {-1, 2, 6, -1, 9, 6, 7};
  hlp2::display(ai3.data(), ai3.data()+ai3.size());
  hlp2::display(ash3.data(), ash3.data()+ash3.size());
  same = hlp2::equal(ai3.data(), ai3.data()+ai3.size(), ash3.data());
  std::cout << "Arrays are " << (same ? "equal\n" : "not equal\n");
}

void TestSum() {
  std::cout << "***** Sum1 *****" << "\n";
  std::array<int, 13> ai {3, 6, 8, 0, -1, 5, -3, -9, 3, 5, 8, -20, -5};
  hlp2::display(ai.data(), ai.data()+ai.size());

  std::cout << "Sum in ai[0,1)  is: " << hlp2::sum(ai.data(),   ai.data()+1) << "\n";
  std::cout << "Sum in ai[0,2)  is: " << hlp2::sum(ai.data(),   ai.data()+2) << "\n";
  std::cout << "Sum in ai[5,7)  is: " << hlp2::sum(ai.data()+5, ai.data()+5+2) << "\n";
  std::cout << "Sum in ai[0,13) is: " << hlp2::sum(ai.data(),   ai.data()+ai.size()) << "\n";

  std::array<double, 13> ad {3.3, 6.4, 8.5, 0.6, -1.7, 5.8, -3.9, -9.0, 3.1, 5.2, 8.3, -20.4, -5.5};
  hlp2::display(ad.data(), ad.data()+ad.size());
  std::cout << "Sum in ad[0,1) is:  " << hlp2::sum(ad.data(),   ad.data()+1) << "\n";
  std::cout << "Sum in ad[0,2) is:  " << hlp2::sum(ad.data(),   ad.data()+2) << "\n";
  std::cout << "Sum in ad[5,7) is:  " << hlp2::sum(ad.data()+5, ad.data()+5+2) << "\n";
  std::cout << "Sum in ad[0,13) is: " << hlp2::sum(ad.data(),   ad.data()+ad.size()) << "\n";

  std::array<std::string, 1> as1 {"learningaboutc++templates"};
  std::string str_tot = hlp2::sum(as1.data(), as1.data()+1);
  std::cout << "Concatenation in as1[0,1) is:   " << str_tot << "\n";

  std::array<std::string, 2> as2 {"learning about C++", " templates"};
  str_tot = hlp2::sum(as2.data(), as2.data()+2);
  std::cout << "Concatenation in as2[0,2) is:   " << str_tot << "\n";

  std::array<std::string, 25> as3 {"a","b","c","d", "e","f","g","h","i","j",
                      "k","l","m","n","o","p","q","r","s","t","u","w","x","y","z"};
  str_tot = hlp2::sum(as3.data(), as3.data()+as3.size());
  std::cout << "Concatenation in as3[0,25) is:  " << str_tot << "\n";

  str_tot = hlp2::sum(as3.data()+24, as3.data()+as3.size());
  std::cout << "Concatenation in as3[24,25) is: " << str_tot << "\n";
}

} // end anonymous namespace
