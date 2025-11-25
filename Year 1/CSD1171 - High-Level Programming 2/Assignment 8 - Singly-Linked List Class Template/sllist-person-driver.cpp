#include "sllist.hpp"
#include <iostream> // cout, endl
#include <cstdio>   // sprintf
#include <cstring>
#include <cstddef>  // size_t
#include <array>

namespace {
  struct Person;
  void FillPersonRecs();
  void FreePersonRecs();

  void TestInitListCtor();      // 1
  void TestArrayCtor();         // 2
  void TestCopyCtorAndSwap();   // 3
  void TestPushFront();         // 4
  void TestPushBack();          // 5
  void TestPushFrontBack();     // 6
  void TestPopFront();          // 7  
  void TestCopyAss();           // 8
  void TestInitListAss();       // 9
  void TestAddAssign();         // 10
  void TestInitListAddAssign(); // 11
  void TestAdd();               // 12
  void TestInitListAdd();       // 13
  void TestSubscript();         // 14

  template <typename T> void InUse();
} // end anonymous namespace 

int main(int argc, char *argv[]) {
  if (argc < 2) {
		std::cout << "Usage: ./listint.out test-number\n";
		std::cout << "Tests 1 through 18 are specified by functions listed above.\n"
							<< "Test 0 will include all the 18 tests described above.\n";
    return 0;
  }

  int const num_of_tests {14};
  int test_num{std::stoi(argv[1])};
	test_num = test_num < 0 ? 0 : test_num;
	test_num = test_num > num_of_tests ? num_of_tests : test_num;

  using TPF = void (*)();
  std::array<TPF, num_of_tests> Tests {
    TestInitListCtor,      // 1
    TestArrayCtor,         // 2
    TestCopyCtorAndSwap,   // 3
    TestPushFront,         // 4
    TestPushBack,          // 5
    TestPushFrontBack,     // 6
    TestPopFront,          // 7
    TestCopyAss,           // 8
    TestInitListAss,       // 9
    TestAddAssign,         // 10
    TestInitListAddAssign, // 11
    TestAdd,               // 12
    TestInitListAdd,       // 13
    TestSubscript          // 14
  };
  
  FillPersonRecs();
  if (test_num == 0) {
		for (TPF x : Tests) {
      x();
    }
  } else {
    Tests[test_num - 1]();
  }
  FreePersonRecs();
}

namespace {

struct Person {
  enum {ID_LEN = 6, NAME_LEN = 25};

  char ID[ID_LEN+1];
  char lastName[NAME_LEN+1];   
  char firstName[NAME_LEN+1];  
  double salary;        
  unsigned years;           
};

struct Person PEOPLE[] = {
  {"101000", "Last",       "First",        0,  0},
  {"101001", "Faith",      "Ian",      80000, 10},
  {"102001", "Tufnel",     "Nigel",    90000, 12},
  {"103001", "Savage",     "Viv",      50000,  4},
  {"104001", "Shrimpton",  "Mick",     50000,  4},
  {"105001", "Besser",     "Joe",      40000,  1},
  {"106001", "Smalls",     "Derek",    80000, 10},
  {"107001", "St.Hubbins", "David",    90000, 12},
  {"108001", "Fleckman",   "Bobbi",   120000,  8},
  {"109001", "Eton-Hogg",  "Denis",   250000, 22},
  {"110001", "Upham",      "Danny",    60000,  5},
  {"111001", "McLochness", "Ross",     60000,  5},
  {"112001", "Pudding",    "Ronnie",   50000,  2},
  {"113001", "Schindler",  "Danny",    60000,  3},
  {"114001", "Pettibone",  "Jeanine",  85000,  3},
  {"115001", "Fame",       "Duke",     95000,  8},
  {"116001", "Fufkin",     "Artie",    45000,  1},
  {"117001", "DiBergi",    "Marty",    15000,  7},
  {"118001", "Floyd",      "Pink",     25000,  6},
  {"119001", "Zeppelin",   "Led",      35000,  3},
  {"120001", "Mason",      "Nick",     15000,  7},
  {"121001", "Wright",     "Richard",  17000,  9},
  {"122001", "Waters",     "Roger",    10000,  3},
  {"123001", "Gilmore",    "David",    19000,  5}
};

Person **PersonRecs;

void FreePersonRecs() {
  size_t count = sizeof(PEOPLE) / sizeof(*PEOPLE);
  for (size_t i = 0; i < count; i++) {
    delete PersonRecs[i];
  }
  delete[] PersonRecs;
}

void FillPersonRecs() {
  size_t count = sizeof(PEOPLE) / sizeof(*PEOPLE);
  PersonRecs = (Person **) new Person[count];

  for (size_t i{}; i < count; i++) {
    Person *pp = new Person;
    std::strcpy(pp->ID, PEOPLE[i].ID);
    std::strcpy(pp->lastName, PEOPLE[i].lastName);
    std::strcpy(pp->firstName, PEOPLE[i].firstName);
    pp->salary = PEOPLE[i].salary;
    pp->years = PEOPLE[i].years;
    PersonRecs[i] = pp;
  }
}

std::ostream& operator<<(std::ostream& os, Person const& rhs) {
  char buffer[1024];
  std::sprintf(buffer, "Key: %8s, Name: %12s, %12s    Salary:%7.0f, Years: %2d\n", 
                  rhs.ID, rhs.lastName, rhs.firstName, rhs.salary, rhs.years);
  os << buffer;
  return os;
}

// construct list from initializer_list
void TestInitListCtor() {
  std::cout << "---------------------TestInitListCtor---------------------\n";

  std::initializer_list<Person> init_list {
              Person{"101001", "Faith",      "Ian",      80000, 10},
              Person{"102001", "Tufnel",     "Nigel",    90000, 12},
              Person{"103001", "Savage",     "Viv",      50000,  4},
              Person{"104001", "Shrimpton",  "Mick",     50000,  4},
              Person{"105001", "Besser",     "Joe",      40000,  1},
              Person{"106001", "Smalls",     "Derek",    80000, 10},
              Person{"107001", "St.Hubbins", "David",    90000, 12},
              Person{"108001", "Fleckman",   "Bobbi",   120000,  8},
              Person{"109001", "Eton-Hogg",  "Denis",   250000, 22},
              Person{"110001", "Upham",      "Danny",    60000,  5}
            };
  hlp2::sllist<Person> list1(init_list);
  std::cout << "list1:\n" << list1;
  std::cout << "Items in list1: " << list1.size() << "\n";
  InUse<Person>();

  std::cout << "clear list1:\n";
  list1.clear();
  std::cout << "Items in list1: " << list1.size() << "\n";
  InUse<Person>();
}

// construct list from from array
void TestArrayCtor() {
  std::cout << "---------------------TestArrayCtor---------------------\n";

  Person const array[] = {{"101001", "Faith",      "Ian",      80000, 10},
                          {"102001", "Tufnel",     "Nigel",    90000, 12},
                          {"103001", "Savage",     "Viv",      50000,  4},
                          {"104001", "Shrimpton",  "Mick",     50000,  4},
                          {"105001", "Besser",     "Joe",      40000,  1},
                          {"106001", "Smalls",     "Derek",    80000, 10},
                          {"107001", "St.Hubbins", "David",    90000, 12},
                          {"108001", "Fleckman",   "Bobbi",   120000,  8},
                          {"109001", "Eton-Hogg",  "Denis",   250000, 22},
                          {"110001", "Upham",      "Danny",    60000,  5}
                        };
  hlp2::sllist<Person> list1(array, array+sizeof(array) / sizeof(*array)); 
  std::cout << "list1:\n" << list1;
  std::cout << "Items in list1: " << list1.size() << "\n";
  InUse<Person>();

  std::cout << "clear list1:\n";
  list1.clear();
  std::cout << "Items in list1: " << list1.size() << "\n";
  InUse<Person>();
}

void TestCopyCtorAndSwap() {
  std::cout << "---------------------TestCopyCtorAndSwap---------------------\n";

  size_t const size{10};
  hlp2::sllist<Person> list1;
  for (size_t i{1}; i <= size; i++) {
    list1.push_front(*PersonRecs[i]);
  }
  std::cout << "list1:\n" << list1;

  hlp2::sllist<Person> list2{list1};
  std::cout << "list2:\n" << list2;

  hlp2::sllist<Person> list3;
  for (size_t i{1}; i <= size; i++) {
    list3.push_back(*PersonRecs[i]);
  }
  std::cout << "list3:\n" << list3;

  hlp2::sllist<Person> const list4{list3};
  std::cout << "list4:\n" << list4;
  std::cout << "list4 is " << (list4.empty() ? "" : "not") << " empty\n";
  std::cout << "Items in list1/list2/list3/list4: "
            << list1.size() << "/" << list2.size() << "/" << list3.size() << "/" << list4.size() << "\n";
  InUse<Person>();

  hlp2::sllist<Person> list5{list4};
  std::cout << "list5:\n" << list5;
  std::cout << "Items in list1/list2/list3/list4/list5: " << list1.size() << "/" << list2.size()
            << "/" << list3.size() << "/" << list4.size() << "/" << list5.size() << "\n";
  InUse<Person>();

//----
  std::cout << "testing member function swap ...\n";
  std::cout << "Before swapping list2 and list5:\n";
  std::cout << "list2:\n" << list2;
  std::cout << "list5:\n" << list5;
  std::cout << "Items in list1/list2/list3/list4/list5: " << list1.size() << "/" << list2.size()
            << "/" << list3.size() << "/" << list4.size() << "/" << list5.size() << "\n";
  InUse<Person>();
  list2.swap(list5);
  std::cout << "After swapping list2 and list5:\n";
  std::cout << "list2:\n" << list2;
  std::cout << "list5:\n" << list5;
  std::cout << "Items in list1/list2/list3/list4/list5: " << list1.size() << "/" << list2.size()
            << "/" << list3.size() << "/" << list4.size() << "/" << list5.size() << "\n";
  InUse<Person>();

  std::cout << "testing non-member function swap ...\n";
  list2 = list1;
  list5 = list4;
  std::cout << "Before swapping list2 and list5:\n";
  std::cout << "list2:\n" << list2;
  std::cout << "list5:\n" << list5;
  std::cout << "Items in list1/list2/list3/list4/list5: " << list1.size() << "/" << list2.size()
            << "/" << list3.size() << "/" << list4.size() << "/" << list5.size() << "\n";
  InUse<Person>();
  hlp2::swap(list2, list5);
  std::cout << "After swapping list2 and list5:\n";
  std::cout << "list2:\n" << list2;
  std::cout << "list5:\n" << list5;
  std::cout << "Items in list1/list2/list3/list4/list5: " << list1.size() << "/" << list2.size()
            << "/" << list3.size() << "/" << list4.size() << "/" << list5.size() << "\n";
  InUse<Person>();

  // -----

  std::cout << "clear list1 and list2:\n";
  list1.clear();
  list2.clear();
  std::cout << "Items in list1/list2/list3/list4: "
            << list1.size() << "/" << list2.size() << "/" << list3.size() << "/" << list4.size() << "\n";
  InUse<Person>();
}

void TestPushFront() {
  std::cout << "---------------------TestPushFront---------------------\n";

  size_t size{5};
  hlp2::sllist<Person> list1;
  for (size_t i{1}; i <= size; i++) {
    list1.push_front(*PersonRecs[i]);
  }
  std::cout << "list1:\n" << list1;
  std::cout << "Items in list1: " << list1.size() << "\n";
  InUse<Person>();

  std::cout << "clear list1:\n";
  list1.clear();
  std::cout << "Items in list1: " << list1.size() << "\n";
  InUse<Person>();
}

void TestPushBack() {
  std::cout << "---------------------TestPushBack---------------------\n";

  size_t size{5};
  hlp2::sllist<Person> list1;
  for (size_t i{1}; i <= size; i++) {
    list1.push_back(*PersonRecs[i]);
  }
  std::cout << "list1:\n" << list1;
  std::cout << "Items in list1: " << list1.size() << "\n";
  InUse<Person>();

  std::cout << "clear list1:\n";
  list1.clear();
  std::cout << "Items in list1: " << list1.size() << "\n";
  InUse<Person>();
}

void TestPushFrontBack() {
  std::cout << "---------------------TestPushFrontBack---------------------\n";

  size_t size{5};
  hlp2::sllist<Person> list1;
  for (size_t i{1}; i <= size; i++) {
    list1.push_front(*PersonRecs[i]);
    list1.push_back(*PersonRecs[i + 10]);
  }
  std::cout << "list1:\n" << list1;
  std::cout << "Items in list1: " << list1.size() << "\n";
  InUse<Person>();

  std::cout << "clear list1:\n";
  list1.clear();
  std::cout << "Items in list1: " << list1.size() << "\n";
  InUse<Person>();
}

void TestPopFront() {
  std::cout << "---------------------TestPopFront---------------------\n";

  size_t size{5};
  hlp2::sllist<Person> list1;
  for (size_t i{1}; i <= size; i++) {
    list1.push_front(*PersonRecs[i]);
  }
  std::cout << "list1:\n" << list1;
  std::cout << "Items in list1: " << list1.size() << "\n";
  InUse<Person>();

  while (!list1.empty()) {
    Person item = list1.front();
    list1.pop_front();
    std::cout << "First item in list1 was: " << item << "\n";
    std::cout << "list1:\n" << list1;
    std::cout << "Items in list1: " << list1.size() << "\n";
  }
  InUse<Person>();

  std::cout << "clear list1:\n";
  list1.clear();
  std::cout << "Items in list1: " << list1.size() << "\n";
  InUse<Person>();
}

void TestCopyAss() {
  std::cout << "---------------------TestCopyAss---------------------\n";

  size_t size{10};
  hlp2::sllist<Person> list1, list2, list3;
  for (size_t i{1}; i <= size; i++) {
    list1.push_front(*PersonRecs[i]);
  }
  std::cout << "Before assignment:" << "\n";
  std::cout << "list 1:\n" << list1;
  std::cout << "list 2:\n" << list2;
  std::cout << "list 3:\n" << list3;
  std::cout << "Items in list1/list2/list3: " << list1.size() << "/" << list2.size()
            << "/" << list3.size() << "\n";
  InUse<Person>();

  list3 = list2 = list1;
  std::cout << "After assignment:" << "\n";
  std::cout << "list 1:\n" << list1;
  std::cout << "list 2:\n" << list2;
  std::cout << "list 3:\n" << list3;
  std::cout << "Items in list1/list2/list3: " << list1.size() << "/" << list2.size()
            << "/" << list3.size() << "\n";
  InUse<Person>();

  list1 = list1;
  std::cout << "After self assignment:" << "\n";
  std::cout << "list 1:\n" << list1 << "\n";
  std::cout << "Items in list1/list2/list3: " << list1.size() << "/" << list2.size()
            << "/" << list3.size() << "\n";
  InUse<Person>();

  std::cout << "clear list1:\n";
  list1.clear();
  std::cout << "Items in list1/list2/list3: " << list1.size() << "/" << list2.size()
            << "/" << list3.size() << "\n";
  InUse<Person>();
}

void TestInitListAss() {
  std::cout << "---------------------TestInitListAss---------------------\n";

  hlp2::sllist<Person> list1, list2, list3;
  size_t size{10};
  for (size_t i{1}; i <= size; i++) {
    list1.push_front(*PersonRecs[i]);
  }
  std::cout << "Before assignment:" << "\n";
  std::cout << "list 1:\n" << list1;
  std::cout << "list 2:\n" << list2;
  std::cout << "list 3:\n" << list3;
  std::cout << "Items in list1/list2/list3: " << list1.size() << "/" << list2.size()
            << "/" << list3.size() << "\n";
  InUse<Person>();

  std::initializer_list<Person> init_list {
              Person{"101001", "Faith",      "Ian",      80000, 10},
              Person{"102001", "Tufnel",     "Nigel",    90000, 12},
              Person{"103001", "Savage",     "Viv",      50000,  4},
              Person{"104001", "Shrimpton",  "Mick",     50000,  4},
              Person{"105001", "Besser",     "Joe",      40000,  1},
              Person{"106001", "Smalls",     "Derek",    80000, 10},
              Person{"107001", "St.Hubbins", "David",    90000, 12},
              Person{"108001", "Fleckman",   "Bobbi",   120000,  8},
              Person{"109001", "Eton-Hogg",  "Denis",   250000, 22},
              Person{"110001", "Upham",      "Danny",    60000,  5}
            };

  list3 = list2 = list1 = init_list;
  std::cout << "After assignment:" << "\n";
  std::cout << "list 1:\n" << list1;
  std::cout << "list 2:\n" << list2;
  std::cout << "list 3:\n" << list3;

  std::cout << "Items in list1/list2/list3: " << list1.size() << "/" << list2.size()
            << "/" << list3.size() << "\n";
  InUse<Person>();

  list1 = list1;
  std::cout << "After self assignment:" << "\n";
  std::cout << "list 1:\n" << list1;
  std::cout << "Items in list1/list2/list3: " << list1.size() << "/" << list2.size()
            << "/" << list3.size() << "\n";
  InUse<Person>();

  std::cout << "clear list1:\n";
  list1.clear();
  std::cout << "Items in list1/list2/list3: " << list1.size() << "/" << list2.size()
            << "/" << list3.size() << "\n";
  InUse<Person>();
}

void TestAddAssign() {
  std::cout << "---------------------TestAddAssign---------------------\n";

  size_t size{5};
  hlp2::sllist<Person> list1, list2;
  for (size_t i{1}; i <= size; i++) {
    list1.push_front(*PersonRecs[i]);
    list2.push_front(*PersonRecs[i + 5]);
  }
  std::cout << "Before addition:" << "\n";
  std::cout << "list 1:\n" << list1;
  std::cout << "list 2:\n" << list2;
  std::cout << "Items in list1/list2: " << list1.size() << "/" << list2.size() << "\n";
  InUse<Person>();

  list1 += list2;
  std::cout << "After addition:" << "\n";
  std::cout << "list 1:\n" << list1;
  std::cout << "list 2:\n" << list2;
  std::cout << "Items in list1/list2: " << list1.size() << "/" << list2.size() << "\n";
  InUse<Person>();
  
  list1.clear();
  list2.clear();
  std::cout << "Items in list1/list2: " << list1.size() << "/" << list2.size() << "\n";
  InUse<Person>();

  for (size_t i = 1; i <= size; i++) {
    list2.push_front(*PersonRecs[i]);
  }
  std::cout << "Before addition:" << "\n";
  std::cout << "list 1:\n" << list1;
  std::cout << "list 2:\n" << list2;
  std::cout << "Items in list1/list2: " << list1.size() << "/" << list2.size() << "\n";
  InUse<Person>();

  list2 += list1;
  std::cout << "After addition:" << "\n";
  std::cout << "list 1:\n" << list1;
  std::cout << "list 2:\n" << list2;
  std::cout << "Items in list1/list2: " << list1.size() << "/" << list2.size() << "\n";
  InUse<Person>();

  list1.clear();
  list2.clear();
  std::cout << "Items in list1/list2: " << list1.size() << "/" << list2.size() << "\n";
  InUse<Person>();

  hlp2::sllist<Person> list3;
  for (size_t i{1}; i <= size; i++) {
    list1.push_front(*PersonRecs[i]);
    list2.push_front(*PersonRecs[i + 8]);
    list3.push_front(*PersonRecs[i + 15]);
  }
  std::cout << "Before addition:" << "\n";
  std::cout << "list 1:\n" << list1;
  std::cout << "list 2:\n" << list2;
  std::cout << "list 3:\n" << list3;
  std::cout << "Items in list1/list2/list3: " << list1.size() << "/" << list2.size()
            << "/" << list3.size() << "\n";
  InUse<Person>();

  list1 += list2 += list3;
  std::cout << "After addition:" << "\n";
  std::cout << "list 1:\n" << list1;
  std::cout << "list 2:\n" << list2;
  std::cout << "list 3:\n" << list3;
  std::cout << "Items in list1/list2/list3: " << list1.size() << "/" << list2.size()
            << "/" << list3.size() << "\n";
  InUse<Person>();

  std::cout << "clear list1:\n";
  list1.clear();
  std::cout << "Items in list1/list2/list3: " << list1.size() << "/" << list2.size()
            << "/" << list3.size() << "\n";
  InUse<Person>();
}

void TestInitListAddAssign() {
  std::cout << "---------------------TestInitListAddAssign---------------------\n";

  size_t size{5};
  hlp2::sllist<Person> list1, list2;
  for (size_t i{1}; i <= size; i++) {
    list1.push_front(*PersonRecs[i]);
    list2.push_front(*PersonRecs[i + 5]);
  }
  std::cout << "Before addition:" << "\n";
  std::cout << "list 1:\n" << list1;
  std::cout << "list 2:\n" << list2;
  std::cout << "Items in list1/list2: " << list1.size() << "/" << list2.size() << "\n";
  InUse<Person>();

  std::initializer_list<Person> init_list {
              Person{"101001", "Faith",      "Ian",      80000, 10},
              Person{"102001", "Tufnel",     "Nigel",    90000, 12},
              Person{"103001", "Savage",     "Viv",      50000,  4},
              Person{"104001", "Shrimpton",  "Mick",     50000,  4},
              Person{"105001", "Besser",     "Joe",      40000,  1},
              Person{"106001", "Smalls",     "Derek",    80000, 10},
              Person{"107001", "St.Hubbins", "David",    90000, 12},
              Person{"108001", "Fleckman",   "Bobbi",   120000,  8},
              Person{"109001", "Eton-Hogg",  "Denis",   250000, 22},
              Person{"110001", "Upham",      "Danny",    60000,  5}
            };
  list1 += list2 += init_list;
  std::cout << "After addition:" << "\n";
  std::cout << "list 1:\n" << list1;
  std::cout << "list 2:\n" << list2;
  std::cout << "Items in list1/list2: " << list1.size() << "/" << list2.size() << "\n";
  InUse<Person>();

  list1.clear();
  list2.clear();
  std::cout << "Items in list1/list2: " << list1.size() << "/" << list2.size() << "\n";
  InUse<Person>();
}

void TestAdd() {
  std::cout << "---------------------TestAdd---------------------\n";

  size_t size{5};
  hlp2::sllist<Person> list1, list2;
  for (size_t i{1}; i <= size; i++) {
    list1.push_front(*PersonRecs[i]);
    list2.push_front(*PersonRecs[i + 10]);
  }
  std::cout << "Before addition:" << std::endl;
  std::cout << "list 1:\n" << list1;
  std::cout << "list 2:\n" << list2;
  std::cout << "Items in list1/list2: " << list1.size() << "/" << list2.size() << "\n";
  InUse<Person>();

  hlp2::sllist<Person> list3 {list1 + list2};
  std::cout << "After addition:" << std::endl;
  std::cout << "ist 1:\n" << list1;
  std::cout << "list 2:\n" << list2;
  std::cout << "list 3:\n" << list3 << "\n";
  std::cout << "Items in list1/list2/list3: " << list1.size() << "/" << list2.size()
            << "/" << list3.size() << "\n";
  InUse<Person>();

  std::cout << "clear list1:\n";
  list1.clear();
  std::cout << "Items in list1/list2/list3: " << list1.size() << "/" << list2.size()
            << "/" << list3.size() << "\n";
  InUse<Person>();
}

void TestInitListAdd() {
  std::cout << "---------------------TestInitListAdd---------------------\n";

  size_t size{5};
  hlp2::sllist<Person> list1, list2;
  for (size_t i{1}; i <= size; i++) {
    list1.push_front(*PersonRecs[i]);
    list2.push_front(*PersonRecs[i + 10]);
  }
  std::cout << "Before addition:" << std::endl;
  std::cout << "list 1:\n" << list1;
  std::cout << "list 2:\n" << list2;
  std::cout << "Items in list1/list2: " << list1.size() << "/" << list2.size() << "\n";
  InUse<Person>();

  std::initializer_list<Person> init_list {
              Person{"101001", "Faith",      "Ian",      80000, 10},
              Person{"102001", "Tufnel",     "Nigel",    90000, 12},
              Person{"103001", "Savage",     "Viv",      50000,  4},
              Person{"104001", "Shrimpton",  "Mick",     50000,  4},
              Person{"105001", "Besser",     "Joe",      40000,  1},
              Person{"106001", "Smalls",     "Derek",    80000, 10},
              Person{"107001", "St.Hubbins", "David",    90000, 12},
              Person{"108001", "Fleckman",   "Bobbi",   120000,  8},
              Person{"109001", "Eton-Hogg",  "Denis",   250000, 22},
              Person{"110001", "Upham",      "Danny",    60000,  5}
  };
  hlp2::sllist<Person> const list3{list1 + list2};
  std::cout << "slist 3:\n" << list3;
  std::cout << "Items in list1/list2/list3: "
            << list1.size() << "/" << list2.size() << "/" << list3.size() << "\n";
  InUse<Person>();

  hlp2::sllist<Person> list4{init_list + list1 + list2 + list3 + init_list};
  std::cout << "After addition:" << "\n";
  std::cout << "list 4:\n" << list4;
  std::cout << "Items in list1/list2/list3/list4: "
            << list1.size() << "/" << list2.size() << "/" << list3.size() << "/" << list4.size() << "\n";
  InUse<Person>();

  std::cout << "clear list1:\n";
  list1.clear();
  std::cout << "Items in list1/list2/list3/list4: "
            << list1.size() << "/" << list2.size() << "/" << list3.size() << "/" << list4.size() << "\n";
  InUse<Person>();
}

void TestSubscript() {
  std::cout << "---------------------TestSubscript---------------------\n";

  size_t size{10};
  hlp2::sllist<Person> list1;
  for (size_t i{1}; i <= size; i++) {
    list1.push_back(*PersonRecs[i]);
  }
  std::cout << "list[0] = " << list1[0];
  std::cout << "list[5] = " << list1[5];
  std::cout << "list[9] = " << list1[9];
  std::cout << "Items in list1: " << list1.size() << "\n";
  InUse<Person>();

  hlp2::sllist<Person> list2;
  for (size_t i = 1; i <= size; i++) {
    list2.push_back(*PersonRecs[i]);
  }
  std::cout << "Original list2:\n" << list2;

  list2[0] = *PersonRecs[12];
  list2[5] = *PersonRecs[15];
  list2[9] = *PersonRecs[19];
  list2[7] = list2[2];
  std::cout << "Modified list2:\n" << list2;
  std::cout << "Items in list1/list2: " << list1.size() << "/" << list2.size() << "\n";
  InUse<Person>();

  std::cout << "clear list1:\n";
  list1.clear();
  std::cout << "Items in list1/list2: " << list1.size() << "/" << list2.size() << "\n";
  InUse<Person>();
}

template <typename T>
void InUse() {
  std::cout << "============================================\n";
  std::cout << "Total number of Lists/Nodes in use: ";
  std::cout << hlp2::sllist<T>::object_count() << "/" << hlp2::sllist<T>::node_count();
  std::cout << "\n============================================\n\n";
}

} // end anonymous namespace
