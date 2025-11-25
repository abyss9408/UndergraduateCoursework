#include <iostream>
#include <iomanip>
#include "sllist.hpp"

// declarations of unit tests and other private functions in anonymous namespace
namespace {
std::ostream& operator<<(std::ostream&, hlp2::sllist const *ptr_sll);
void incr(hlp2::sllist *ptr_sll, int adder);
void test_find();
void test_push();
void test_remove();
void test_insert();
}

// this is the purpose of the driver - to call each of these tests in turn ...
int main() {
  test_push();
  test_find();
  test_insert();
  test_remove();
}

// private functions ...
namespace {
// overload operator<< to write values of elements in type sllist
std::ostream& operator<<(std::ostream& os, hlp2::sllist const *ptr_sll) {
  if (hlp2::empty(ptr_sll)) {
    return os;
  }
  hlp2::node const *head {hlp2::front(ptr_sll)};
  while (head) {
    os << hlp2::data(head) << std::setw(3) << " ";
    head = hlp2::next(head);
  }
  return os;
}

// increment value of each element of linked list with parameter adder
void incr(hlp2::sllist *ptr_sll, int adder) {
  if (hlp2::empty(ptr_sll)) {
    return;
  }
  hlp2::node *head {hlp2::front(ptr_sll)};
  while (head) {
    hlp2::data(head, adder+data(head));
    head = hlp2::next(head);
  }
}

void test_push() {
  using namespace hlp2;
  using std::setw, std::cout;

  cout << "******************** Push ********************\n";

  sllist *list1 {construct()}, *list2 {construct()};
  for (int x : {12,34,21,56,38,94,23,22,67,56,88,19,59,10,17}) {
    push_back(list1, x);
    push_front(list2, x);
    cout << "List1 (" << setw(2) << size(list1) << " nodes): " << list1 << "\n";
    cout << "List2 (" << setw(2) << size(list2) << " nodes): " << list2 << "\n";
  }

  incr(list1, 5); // increment every element in list1 by 5
  cout << "List1 (" << setw(2) << size(list1) << " nodes): " << list1 << "\n";
  incr(list2, 3); // increment every element in list2 by 3
  cout << "List2 (" << setw(2) << size(list2) << " nodes): " << list2 << "\n";

  destruct(list2); // accessing list interface thro' list2 is now unsafe ... 
  destruct(list1); // accessing list interface thro' list1 is now unsafe ... 
}

void test_find() {
  using namespace hlp2;
  using std::setw, std::cout;

  cout << "******************** Find ********************\n";

  sllist *list {construct()};
  for (int x : {22,23,38,94,56,34,67,56,88,19,10,59}) {
    push_back(list, x);
  }
  cout << "List (" << setw(2) << size(list) << " nodes): " << list << "\n";

  // Find an existing value
  int val = 56;
  node *node = find(list, val);
  cout << "Value " << (node?data(node):val) << (node?"":" not") << " found.\n";

  val = 22;
  node = find(list, val);
  cout << "Value " << (node?data(node):val) << (node?"":" not") << " found.\n";

  val = 59;
  node = find(list, val);
  cout << "Value " << (node?data(node):val) << (node?"":" not") << " found.\n";

  val = 123;
  node = find(list, val);
  cout << "Value " << (node?data(node):val) << (node?"":" not") << " found.\n";

  destruct(list); // accessing list interface thro' list is now unsafe ... 
}

void test_insert() {
  std::cout << "******************** Insert ********************\n";

  hlp2::sllist *list {hlp2::construct()};
  for (int x : {12, 34, 21, 56, 38}) {
    hlp2::push_back(list, x);
  }
  std::cout << "List (" << std::setw(2) << hlp2::size(list) << " nodes): " << list << "\n";

  std::cout << "inserting... 91 at position 3\n";
  hlp2::insert(list, 91, 3);
  std::cout << "List (" << std::setw(2) << hlp2::size(list) << " nodes): " << list << "\n";

  std::cout << "inserting... 92 at position 0\n";
  hlp2::insert(list, 92, 0);
  std::cout << "List (" << std::setw(2) << hlp2::size(list) << " nodes): " << list << "\n";

  std::cout << "inserting... 93 at position 50\n";
  hlp2::insert(list, 93, 50);
  std::cout << "List (" << std::setw(2) << hlp2::size(list) << " nodes): " << list << "\n";
 
  std::cout << "inserting... 94 at position 99\n";
  hlp2::insert(list, 94, 99);
  std::cout << "List (" << std::setw(2) << hlp2::size(list) << " nodes): " << list << "\n";
  
  hlp2::destruct(list); // accessing list interface thro' list is now unsafe ... 
}

void test_remove() {
  using namespace hlp2;
  using std::setw, std::cout;

  std::cout << "******************** Remove ********************\n";

  sllist *list {construct()};
  for (int x : {22, 23, 38, 94, 56, 34, 67, 56, 88, 19, 10, 59}) {
    push_back(list, x);
  }
  cout << "List (" << setw(2) << size(list) << " nodes): " << list << "\n";

  remove_first(list, 56);
  cout << "Deleting: 56 (" << setw(2) << size(list) << " nodes): " << list << "\n";
  remove_first(list, 22);
  cout << "Deleting: 22 (" << setw(2) << size(list) << " nodes): " << list << "\n";
  remove_first(list, 59);
  cout << "Deleting: 59 (" << setw(2) << size(list) << " nodes): " << list << "\n";
  remove_first(list, 99);
  cout << "Deleting: 99 (" << setw(2) << size(list) << " nodes): " << list << "\n";
  
  for (int x : {23, 38, 94, 34, 67, 56, 88, 19, 10}) {
    remove_first(list, x);
    cout << "Deleting: " << x << " (" << setw(2) << size(list) << " nodes): " << list << "\n";
  }
  
  remove_first(list, 99);
  cout << "Deleting: 99" << " (" << setw(2) << size(list) << " nodes): " << list << "\n";
  
  destruct(list); // accessing list interface thro' list is now unsafe ... 
}

} // end anonymous namespace
