// HLP2
// Culmination of discussion on class design and implementation.
// The Str class is our [much simpler] version of std::string that encapsulates
// a C-style null-terminated string.
// It contains all the features discussed in lectures: data encapsulation, data
// abstraction, standard library compatible types, constructors, destructor,
// single-argument conversion constructor, initializer_list constructor,
// copy-swap idiom for implementing copy assignment operator overload operator,
// operator overloads, overload of op<< and op>>, static data members and member
// functions ...

#ifndef MY_STR_HPP
#define MY_STR_HPP

#include <cstddef>          // for size_t
#include <iostream>         // for std::ostream
#include <initializer_list> // for std::initializer_list

namespace hlp2 {

class Str {
public:
  using value_type             = char;
  using size_type              = size_t;
  using reference              = value_type&;
  using const_reference        = const value_type&;
  using pointer                = value_type*;
  using const_pointer          = const value_type*;
  using iterator               = pointer;
  using const_iterator         = const_pointer;
  using reverse_iterator       = pointer;
  using const_reverse_iterator = const_pointer;

public:
  Str();
  Str(const_pointer);
  Str(Str const&);
  Str(std::initializer_list<char>);
  ~Str();

  Str& operator=(Str const&);
  Str& operator=(const_pointer);
  Str& operator+=(Str const&);
  Str& operator+=(const_pointer);
  reference operator[](size_type);
  const_reference operator[](size_type) const;

  size_type size() const;
  bool empty() const;
  const_pointer c_str() const;
  int compare(Str const& rhs) const;
  void swap(Str&);

  iterator               begin();
  const_iterator         begin() const;
  iterator               end();
  const_iterator         end() const;
  const_iterator         cbegin() const;
  const_iterator         cend() const;

  static size_type       get_count();
private:
  size_type len;
  pointer   data;
  static size_type counter;
};

Str operator+(Str const&, Str const&);
Str operator+(Str const&, Str::const_pointer);
Str operator+(Str::const_pointer, Str const&);

bool operator==(Str const&, Str const&);
bool operator!=(Str const&, Str const&);
bool operator< (Str const&, Str const&);
bool operator<=(Str const&, Str const&);
bool operator> (Str const&, Str const&);
bool operator>=(Str const&, Str const&);

std::ostream& operator<<(std::ostream&, Str const&);
std::istream& operator>>(std::istream&, Str&);

} // endif namespace hlp2

#endif
