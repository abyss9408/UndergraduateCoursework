// HLP2
// Culmination of discussion on class design and implementation.
// Implementation of interface described in str.hpp  ...

#include "str.hpp"
#include <cstring>
#include <iostream>
#include <algorithm>
#include <initializer_list>

namespace hlp2 {

Str::size_type Str::counter {0};

Str::Str() : len{0}, data{new char [len+1]} {
#ifdef DEBUG
  std::cout << __PRETTY_FUNCTION__ << "\n";
#endif
  data[0] = '\0';
  ++counter;
}

Str::Str(Str::const_pointer rhs)
: len{std::strlen(rhs)}, data{new char [len+1]} {
#ifdef DEBUG
  std::cout << __PRETTY_FUNCTION__ << rhs << "\n";
#endif
  std::strcpy(data, rhs);
  ++counter;
}

Str::Str(Str const& rhs)
: len{rhs.len}, data{new char [len+1]} {
#ifdef DEBUG
  std::cout << __PRETTY_FUNCTION__ << rhs.data << "\n";
#endif
  std::strcpy(data, rhs.data);
  ++counter;
}

Str::Str(std::initializer_list<char> rhs)
: len{rhs.size()}, data{new char [len+1]} {
  pointer tmp{data};
  for (char ch : rhs) {
    *tmp++ = ch;
  }
  *tmp = '\0';
#ifdef DEBUG
  std::cout << __PRETTY_FUNCTION__ << data << "\n";
#endif
  ++counter;
}


Str::~Str() {
#ifdef DEBUG
  std::cout << __PRETTY_FUNCTION__ << data << "\n";
#endif
  delete [] data;
  --counter;
}

Str& Str::operator=(Str const& rhs) {
  Str tmp{rhs};
  swap(tmp);
  return *this;
}

Str& Str::operator=(Str::const_pointer rhs) {
  Str tmp{rhs};
  swap(tmp);
  return *this;
}

Str& Str::operator+=(Str const& rhs) {
  size_type tmp_len { len + rhs.len };
  pointer   tmp_data { new value_type [tmp_len + 1]};
  std::strcpy(tmp_data, data);
  std::strcat(tmp_data, rhs.data);

  len = tmp_len;
  delete [] data; data = tmp_data;

  return *this;
}

Str& Str::operator+=(Str::const_pointer rhs) {
  size_type tmp_len { len + std::strlen(rhs)};
  pointer   tmp_data { new value_type [tmp_len + 1]};
  std::strcpy(tmp_data, data);
  std::strcat(tmp_data, rhs);

  len = tmp_len;
  delete [] data; data = tmp_data;

  return *this;
}

Str::reference Str::operator[](Str::size_type index) {
  //return data[index];
  return const_cast<reference>((static_cast<Str const&>(*this))[index]);
}

Str::const_reference Str::operator[](Str::size_type index) const {
  return data[index];
}

Str::size_type Str::size() const {
  return len;
}

bool Str::empty() const {
  return len == 0;
}

Str::const_pointer Str::c_str() const {
  return data;
}

int Str::compare(Str const& rhs) const {
  return std::strcmp(data, rhs.data);
}

void Str::swap(Str& rhs) {
  std::swap(len, rhs.len);
  std::swap(data, rhs.data);
}

Str::iterator Str::begin() {
  return data;
}
Str::const_iterator Str::begin() const {
  return data;
}
Str::iterator Str::end() {
  return data+len;
}
Str::const_iterator Str::end() const {
  return data+len;
}
Str::const_iterator Str::cbegin() const {
  return data;
}
Str::const_iterator Str::cend() const {
  return data+len;
}

Str operator+(Str const& lhs, Str const& rhs) {
  Str tmp {lhs};
  tmp += rhs;
  return tmp;
}

Str operator+(Str const& lhs, Str::const_pointer rhs) {
  Str tmp {lhs};
  tmp += rhs;
  return tmp;
}

Str operator+(Str::const_pointer lhs, Str const& rhs) {
  Str tmp {lhs};
  tmp += rhs;
  return tmp;
}

bool operator==(Str const& lhs, Str const& rhs) {
  return lhs.compare(rhs)==0 ? true : false;
}
bool operator!=(Str const& lhs, Str const& rhs) {
  return !(lhs == rhs);
}
bool operator< (Str const& lhs, Str const& rhs) {
  return lhs.compare(rhs) < 0 ? true : false;
}
bool operator<=(Str const& lhs, Str const& rhs) {
 return lhs.compare(rhs) <= 0 ? true : false;
}
bool operator> (Str const& lhs, Str const& rhs) {
  return lhs.compare(rhs) > 0 ? true : false;
}
bool operator>=(Str const& lhs, Str const& rhs) {
 return lhs.compare(rhs) >= 0 ? true : false;
}

std::ostream& operator<<(std::ostream& os, Str const& rhs) {
  os << rhs.c_str();
  return os;
}

std::istream& operator>>(std::istream& is, Str& rhs) {
  int const max_buff_size {1024};
  Str::value_type buffer[max_buff_size];
  is.width(max_buff_size-1);
  is >> buffer;
  rhs = buffer;
  return is;
}

Str::size_type Str::get_count() {
  return Str::counter;
}

} // end namespace hlp2