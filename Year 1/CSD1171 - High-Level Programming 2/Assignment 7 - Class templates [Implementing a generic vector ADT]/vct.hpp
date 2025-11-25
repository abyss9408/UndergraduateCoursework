/*!************************************************************************
    \file vct.hpp

    \author Bryan Ang Wei Ze

    \par DP email: bryanweize.ang\@digipen.edu

    \par Course: CSD1171 High-Level Programming 2

    \par Programming Assignment #7

    \date 03-09-2024
    
    \brief
    This header file declares class vector and member functions that
    emulate the behaviour of std::vector
**************************************************************************/
////////////////////////////////////////////////////////////////////////////////
#ifndef VCT_HPP
#define VCT_HPP
////////////////////////////////////////////////////////////////////////////////
#include <cstddef>          // need this for size_t
#include <initializer_list> // need this for std::initializer_list<T>
#include <algorithm> // need this for std::swap
namespace hlp2 {

template <typename T>
class vector {
public:
  using size_type = size_t;
  using value_type = T;
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
  vector(std::initializer_list<T> rhs);

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
  vector<T>& operator=(const vector<T> &rhs);

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
  vector<T>& operator=(std::initializer_list<T> rhs);

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
 Removes the last element by decrementing its size
*/
/**************************************************************************/
  void pop_back();

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
 Exchanges contents of object with contents of rhs


\param rhs
 Another hlp2::vector object of same type
*/
/**************************************************************************/
  void swap(vector<T>& rhs);

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

template <typename T>
vector<T>::vector() 
: sz{0}, space{0}, allocs{0}, data{nullptr}
{
    
}

template <typename T>
vector<T>::vector(size_type n) 
: sz{n}, space{n}, allocs{1}, data{new value_type[sz]}
{
    for (size_t i{}; i < sz; ++i)
    {
        data[i] = 0;
    }
}

template <typename T>
vector<T>::vector(std::initializer_list<T> rhs) 
: sz{rhs.size()}, space{rhs.size()}, allocs{1}, data{new value_type[sz]}
{
    pointer tmp{data};
    for (const T &val : rhs)
    {
        *tmp++ = val;
    }
}

template <typename T>
vector<T>::vector(const vector& rhs) 
: sz{rhs.sz}, space{rhs.sz}, allocs{1}, data{new value_type[sz]}
{
    for (size_t i{}; i < sz; ++i)
    {
         data[i] = rhs.data[i];
    }
}

template <typename T>
vector<T>::~vector()
{
    delete[] data;
}

template <typename T>
vector<T>& vector<T>::operator=(const vector<T> &rhs)
{
    vector<T> tmp{rhs};
    swap(tmp);
    std::swap(allocs, tmp.allocs);
    ++allocs;
    return *this;
}

template <typename T>
vector<T>& vector<T>::operator=(std::initializer_list<T> rhs)
{
    vector<T> tmp{rhs};
    swap(tmp);
    std::swap(allocs, tmp.allocs);
    ++allocs;
    return *this;
}

template <typename T>
typename vector<T>::reference vector<T>::operator[](size_type index)
{
    return data[index];
}

template <typename T>
typename vector<T>::const_reference vector<T>::operator[](size_type index) const
{
    return data[index];
}

template <typename T>
typename vector<T>::pointer vector<T>::begin()
{
    return data;
}

template <typename T>
typename vector<T>::pointer vector<T>::end()
{
    return data+sz;
}

template <typename T>
typename vector<T>::const_pointer vector<T>::begin() const
{
    return data;
}

template <typename T>
typename vector<T>::const_pointer vector<T>::end() const
{
    return data+sz;
}

template <typename T>
typename vector<T>::const_pointer vector<T>::cbegin() const
{
    return data;
}

template <typename T>
typename vector<T>::const_pointer vector<T>::cend() const
{
    return data+sz;
}

template <typename T>
void vector<T>::reserve(size_type newsize)
{
    if (newsize <= space)
    {
        return;
    }

    pointer ptr = new value_type[newsize];
    for (size_type i{}; i < sz; ++i)
    {
        ptr[i] = data[i];
    }
    delete[] data;
    data = ptr;
    space = newsize;
    ++allocs;
}

template <typename T>
void vector<T>::resize(size_type newsize)
{
    if (newsize > space)
    {
        reserve(newsize);
        sz = newsize;
    }
    else if (newsize > sz && newsize <= space)
    {
        sz = newsize;
    }
    else if (newsize == sz)
    {
        return;
    }
    else if (newsize < sz)
    {
        sz = newsize;
    }
        
}

template <typename T>
void vector<T>::push_back(value_type value)
{
    if (space == 0)
    {
        reserve(1);
    }
    else if (sz == space)
    {
        reserve(2*space);
    }
        
    data[sz] = value;
    ++sz;
}

template <typename T>
void vector<T>::pop_back()
{
    --sz;
}

template <typename T>
void vector<T>::swap(vector<T>& rhs)
{
    std::swap(sz, rhs.sz);
    std::swap(allocs, rhs.allocs);
    std::swap(space, rhs.space);
    std::swap(data, rhs.data);
}
    
template <typename T>
bool vector<T>::empty() const
{
    if (sz == 0)
    {
        return true;
    }
        
    return false;
}

template <typename T>
typename vector<T>::size_type vector<T>::size() const
{
    return sz;
}

template <typename T>
typename vector<T>::size_type vector<T>::capacity() const
{
    return space;
}

template <typename T>
typename vector<T>::size_type vector<T>::allocations() const
{
    return allocs;
}

} // namespace hlp2

#endif // VCT_HPP
