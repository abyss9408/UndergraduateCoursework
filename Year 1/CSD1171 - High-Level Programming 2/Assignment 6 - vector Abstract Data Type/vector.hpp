/*!************************************************************************
    \file vector.hpp

    \author Bryan Ang Wei Ze

    \par DP email: bryanweize.ang\@digipen.edu

    \par Course: CSD1171 High-Level Programming 2

    \par Programming Assignment #6

    \date 02-28-2024
    
    \brief
    This header file declares class vector and member functions that
    emulate the behaviour of std::vector
**************************************************************************/

////////////////////////////////////////////////////////////////////////////////
#ifndef VECTOR_HPP
#define VECTOR_HPP
////////////////////////////////////////////////////////////////////////////////
#include <cstddef>          // need this for size_t
#include <initializer_list> // need this for std::initializer_list<int>
// read the specs to know which standard library headers cannot be included!!!
 
namespace hlp2 {
	
class vector {
public:
  using size_type = size_t;
  using value_type = int;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using reference = value_type&;
  using const_reference = const value_type&;

public:
/***************************************************************************/
/*!
\brief
 Default constructor
*/
/**************************************************************************/
  vector();

/***************************************************************************/
/*!
\brief
 Single-argument conversion constructor that allocates memory for n elements
 with each element initialized to 0

\param n
 Number of elements
*/
/**************************************************************************/
  explicit vector(size_type n);

/***************************************************************************/
/*!
\brief
 Another single-argument conversion constructor that allocates memory to store
 values in an initializer_list<int>

\param rhs
 initializer_list<int>
*/
/**************************************************************************/
  vector(std::initializer_list<int> rhs);

/***************************************************************************/
/*!
\brief
 Copy constructor

\param rhs
 vector object to be copy constructed from
*/
/**************************************************************************/
  vector(const vector& rhs);

/***************************************************************************/
/*!
\brief
 Destructor
*/
/**************************************************************************/
  ~vector();

/***************************************************************************/
/*!
\brief
 Copy assignment for vector object

\param rhs
 vector object to be copy assigned from

\return
 Reference to the invoking vector object
*/
/**************************************************************************/
  vector& operator=(const vector &rhs);

/***************************************************************************/
/*!
\brief
 Copy assignment for std::initializer_list<int>

\param rhs
 std::initializer_list<int> to be copy assigned from

\return
 Reference to the invoking vector object
*/
/**************************************************************************/
  vector& operator=(std::initializer_list<int> rhs);

/***************************************************************************/
/*!
\brief
 Subscript operator

\param i
 index

\return
 Reference to element wtih index i
*/
/**************************************************************************/
  reference operator[](size_type i);

/***************************************************************************/
/*!
\brief
 Subscript operator

\param i
 index

\return
 const Reference to element wtih index i
*/
/**************************************************************************/
  const_reference operator[](size_type i) const;

/***************************************************************************/
/*!
\brief
 Add element at the end

\param value
 Value to be added
*/
/**************************************************************************/
  void push_back(value_type value);

/***************************************************************************/
/*!
\brief
 Request a change in capacity

\param newsize
 Minimum capacity for the vector
*/
/**************************************************************************/
  void reserve(size_type newsize);

/***************************************************************************/
/*!
\brief
 Change size

\param newsize
 New container size, expressed in number of elements
*/
/**************************************************************************/
  void resize(size_type newsize);

/***************************************************************************/
/*!
\brief
 Test whether vector is empty

\return
 true if the container size is 0, false otherwise
*/
/**************************************************************************/
  bool empty() const;

/***************************************************************************/
/*!
\brief
 Return size

\return
 The number of elements in the container
*/
/**************************************************************************/
  size_type size() const;

/***************************************************************************/
/*!
\brief
 Return size of allocated storage capacity

\return
 The size of the currently allocated storage capacity in the vector
*/
/**************************************************************************/
  size_type capacity() const;

/***************************************************************************/
/*!
\brief
 Return number of allocations done in the vector

\return
 The number of allocations done in the vector
*/
/**************************************************************************/
  size_type allocations() const;

/***************************************************************************/
/*!
\brief
 Return iterator to beginning

\return
 An iterator to the beginning of the vector
*/
/**************************************************************************/
  pointer begin();

/***************************************************************************/
/*!
\brief
 Return iterator to end

\return
 An iterator to the element past the end of the sequence
*/
/**************************************************************************/
  pointer end();

/***************************************************************************/
/*!
\brief
 Return const_iterator to beginning

\return
 A const iterator to the beginning of the vector
*/
/**************************************************************************/
  const_pointer begin() const;

/***************************************************************************/
/*!
\brief
 Return const_iterator to end

\return
 A const iterator to the element past the end of the sequence
*/
/**************************************************************************/
  const_pointer end() const;

/***************************************************************************/
/*!
\brief
 Return const_iterator to beginning

\return
 A const iterator to the beginning of the vector
*/
/**************************************************************************/
  const_pointer cbegin() const;

/***************************************************************************/
/*!
\brief
 Return const_iterator to end

\return
 A const iterator to the element past the end of the sequence
*/
/**************************************************************************/
  const_pointer cend() const;

private:
  size_type sz;     // the number of elements in the array
  size_type space;  // the allocated size (in terms of elements) of the array
  size_type allocs; // number of times space has been updated
  pointer   data;   // the dynamically allocated array
};

} // namespace hlp2

#endif // VECTOR_HPP
