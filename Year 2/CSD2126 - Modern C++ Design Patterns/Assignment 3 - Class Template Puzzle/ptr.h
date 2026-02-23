/*!************************************************************************
  \file ptr.h

  \author Bryan Ang Wei Ze

  \par DP email: bryanweize.ang\@digipen.edu

  \par Course: CSD2126 Modern C++ Design Patterns

  \par Programming Assignment #3

  \date 04-10-2024
  
  \brief
  This header file containing definition of class template Ptr and 
  definitions of member functions of class template Ptr.
**************************************************************************/

#include <iostream> // std::ostream

// TODO: Don't include any other C and C++ standard library headers!!!

#ifndef PTR_H
#define PTR_H

namespace HLP3 {

// partially defined class template Ptr
template <typename T>
class Ptr {
private:
  T *p;                           // DO NOT AMEND!!!
public:
  Ptr()                 = delete; // DO NOT AMEND!!!
  Ptr(Ptr&&)            = delete; // DO NOT AMEND!!!
  Ptr& operator=(Ptr&&) = delete; // DO NOT AMEND!!!
  explicit Ptr(T* _p) : p{_p} {}  // DO NOT AMEND!!!
  ~Ptr() { std::cout << __PRETTY_FUNCTION__ << std::endl;  delete p; } // DO NOT AMEND!!!

  T* get() const { return p; }    // DO NOT AMEND!!!

/*
Your task is to extend this partially defined class template Ptr with six (6) member functions
that will enable the unit tests in driver source file ptr-driver.cpp to successfully compile and
link to generate an executable file that will then generate the correct output for each unit test.

Carefully read the unit tests in driver source file ptr-driver.cpp to determine the member functions that 
you must declare and define for class template Ptr.
The driver source file contains the expected output and provides hints to the member functions that you
must implement to successfully complete the puzzle.

If you don't implement the required functions [again there are six such functions], it is possible to
obtain the same results as described in the driver. However, in such cases, the online grader will
reject your submission.

You can define the member functions inline or outside the class template definition!!!
*/

  Ptr(const Ptr& rhs)
    : p(new T)
  {
    *p = *(rhs.get());
  }

  template <typename U>
  Ptr(const Ptr<U>& rhs)
    : p(new T)
  {
    *p = *(rhs.get());
  }

  Ptr& operator=(const Ptr& rhs)
  {
    *p = *(rhs.get());
    return *this;
  }

  template <typename U>
  Ptr& operator=(const Ptr<U>& rhs)
  {
    *p = *(rhs.get());
    return *this;
  }

  T& operator*()
  {
    return *p;
  }

  T* operator->()
  {
    return p;
  }
};

} // end namespace HLP3

#endif // #ifndef PTR_H
