/*!************************************************************************
    \file ft.hpp

    \author Bryan Ang Wei Ze

    \par DP email: bryanweize.ang\@digipen.edu

    \par Course: CSD1171 High-Level Programming 2

    \par Lab #6

    \date 03-01-2024
    
    \brief
    This header file declares and defines function templates that work on
    containers of different data types
**************************************************************************/

//-------------------------------------------------------------------------
#ifndef FT_H
#define FT_H
//-------------------------------------------------------------------------
#include <iostream>

namespace hlp2
{
/***************************************************************************/
/*!
\brief
 Swaps two objects. There is no return value but the two objects are
 swapped in place.

\param lhs
  Reference to the first object to swap.

\param rhs
 Reference to the second object to swap.
*/
/**************************************************************************/
template <typename T> void swap(T &lhs, T &rhs);

/***************************************************************************/
/*!
\brief
 Displays all elements in a container

\param range_start
 Pointer to the first element of container

\param range_end
 Pointer to one past the last element of container
*/
/**************************************************************************/
template <typename T> void display(T *range_start, T *range_end);

/***************************************************************************/
/*!
\brief
 Swap elements of two ranges

\param range1_start
  Pointer to the first element of first range

\param range1_end
 Pointer to one past the last element of first range

\param range2_start
 Pointer to the first element of second range
*/
/**************************************************************************/
template <typename T> void swap_ranges(T *range1_start, T *range1_end, T *range2_start);

/***************************************************************************/
/*!
\brief
 Remove value from range

\param range_start
  Pointer to the first element of range

\param range_end
 Pointer to one past the last element of range

\param value
 Elements with value to be removed

\return
 Pointer to one past the last element of new range
*/
/**************************************************************************/
template <typename T> T* remove(T *range_start, T *range_end, const T &value);

/***************************************************************************/
/*!
\brief
 Count appearances of value in range

\param range_start
  Pointer to the first element of range

\param range_end
 Pointer to one past the last element of range

\param value
 Value to match

\return
 The number of elements in the range that compare equal to value
*/
/**************************************************************************/
template <typename T1, typename T2> int count(T1 *range_start, T1 *range_end, const T2 &value);

/***************************************************************************/
/*!
\brief
 Find value value in range

\param range_start
 Pointer to the first element of range

\param range_end
 Pointer to one past the last element of range

\param value
 Value to search for

\return
 Pointer to the first element in the range that compares equal to value
*/
/**************************************************************************/
template <typename T> T* find(const T *range_start, const T *range_end, const T &value);

/***************************************************************************/
/*!
\brief
 Copy range of elements

\param range1_start
 Pointer to the first element of range 1

\param range1_end
 Pointer to one past the last element of range 2

\param range2_start
 Pointer to the first element of range 2

\return
 Pointer to one past the last element of range 2
*/
/**************************************************************************/
template <typename T> T* copy(const T *range1_start, const T *range1_end, T *range2_start);

/***************************************************************************/
/*!
\brief
 Fill range with value

\param range1_start
 Pointer to the first element of range

\param range1_end
 Pointer to one past the last element of range

\param value
 Value to assign to the elements in the filled range
*/
/**************************************************************************/
template <typename T1, typename T2> void fill(T1 *range_start, T1 *range_end, const T2 &value);

/***************************************************************************/
/*!
\brief
 Replace value in range

\param range1_start
 Pointer to the first element of range

\param range1_end
 Pointer to one past the last element of range

\param old_value
 Value to be replaced

\param new_value
 Replacement value
*/
/**************************************************************************/
template <typename T1, typename T2> void replace(T1 *range_start, T1 *range_end, const T2 &old_value, const T2 &new_value);

/***************************************************************************/
/*!
\brief
 Return smallest element in range

\param range_start
 Pointer to the first element of range

\param range_end
 Pointer to one past the last element of range

\return
 Pointer to smallest value in the range
*/
/**************************************************************************/
template <typename T> T* min_element(T *range_start, T *range_end);

/***************************************************************************/
/*!
\brief
 Return largest element in range

\param range_start
 Pointer to the first element of range

\param range_end
 Pointer to one past the last element of range

\return
 Pointer to largest value in the range
*/
/**************************************************************************/
template <typename T> T* max_element(T *range_start, T *range_end);

/***************************************************************************/
/*!
\brief
 Test whether the elements in two ranges are equal

\param range1_start
 Pointer to the first element of range 1

\param range1_end
 Pointer to one past the last element of range 1

\param range2_start
 Pointer to the first element of range 2

\return
 true if all the elements in the range 1 compare equal to those of range 2, and
 false otherwise.
*/
/**************************************************************************/
template <typename T1, typename T2> bool equal(T1 *range1_start, T1 *range1_end, T2 *range2_start);

/***************************************************************************/
/*!
\brief
 Sum all values in range

\param range_start
 Pointer to the first element of range

\param range_end
 Pointer to one past the last element of range

\return
 The result of the sum of all the elements in the range
*/
/**************************************************************************/
template <typename T> T sum(T *range_start, T *range_end);

template <typename T>
void swap(T &lhs, T &rhs)
{
  T tmp{lhs};
  lhs = rhs;
  rhs = tmp;
}

template <typename T>
void display(T *start, T*end)
{
  size_t i{};
  size_t size = end - start;
  if (size != 0)
  {
    for (i = 0; i < size - 1; ++i)
    {
        std::cout<< start[i] <<", ";
    }
    std::cout<< start[i] <<'\n';
  }
  else
  {
    std::cout<< '\n';
  }
}

template <typename T>
void swap_ranges(T *range1_start, T *range1_end, T *range2_start)
{
  size_t size = range1_end - range1_start;
  for (size_t i = 0; i < size; ++i)
  {
    T tmp{range1_start[i]};
    range1_start[i] = range2_start[i];
    range2_start[i] = tmp;
  }
}

template <typename T>
T* remove(T *range_start, T *range_end, const T &value)
{
  T*  result = range_start;
  size_t size = range_end - range_start;
  for (size_t i = 0; i < size; ++i)
  {
    if (range_start[i] != value)
    {
      if (result!=range_start+i)
      {
        *result = range_start[i];
      } 
      ++result;
    }
  }

  return result;

}

template <typename T1, typename T2>
int count(T1 *range_start, T1 *range_end, const T2 &value)
{
  size_t size = range_end - range_start;
  size_t count{};
  for (size_t i = 0; i < size; ++i)
  {
    if (range_start[i] == value)
    {
      ++count;
    }
  }

  return count;
}

template <typename T>
T* find(const T *range_start, const T *range_end, const T &value)
{
  size_t i;
  size_t size = range_end - range_start;
  for (i = 0; i < size; ++i)
  {
    if (range_start[i] == value)
    {
      break;
    }
  }

  return const_cast<T*>(range_start) + i;
}

template <typename T>
T* copy(const T *range1_start, const T *range1_end, T *range2_start)
{
  size_t i;
  size_t size = range1_end - range1_start;
  for (i = 0; i < size; ++i)
  {
    range2_start[i] = range1_start[i];
  }

  return range2_start + i;
}

template <typename T1, typename T2>
void fill(T1 *range_start, T1 *range_end, const T2 &value)
{
  size_t size = range_end - range_start;
  for (size_t i = 0; i < size; ++i)
  {
    range_start[i] = value;
  }
}

template <typename T1, typename T2>
void replace(T1 *range_start, T1 *range_end, const T2 &old_value, const T2 &new_value)
{
  size_t size = range_end - range_start;
  for (size_t i = 0; i < size; ++i)
  {
    if (range_start[i] == old_value)
    {
      range_start[i] = new_value;
    }
  }
}

template <typename T>
T* min_element(T *range_start, T *range_end)
{
  T min = range_start[0];
  size_t index{};
  size_t size = range_end - range_start;
  for (size_t i = 1; i < size; ++i)
  {
    if (range_start[i] < min)
    {
      min = range_start[i];
      index = i;
    }
  }
  return range_start+index;
}

template <typename T>
T* max_element(T *range_start, T *range_end)
{
  T max = range_start[0];
  size_t index{};
  size_t size = range_end - range_start;
  for (size_t i = 1; i < size; ++i)
  {
    if (max < range_start[i])
    {
      max = range_start[i];
      index = i;
    }
  }
  return range_start+index;
}

template <typename T1, typename T2>
bool equal(T1 *range1_start, T1 *range1_end, T2 *range2_start)
{
  size_t size = range1_end - range1_start;
  for (size_t i = 0; i < size; ++i)
  {
    if (range1_start[i] != range2_start[i])
    {
      return false;
    }
  }

  return true;
}

template <typename T>
T sum(T *range_start, T *range_end)
{
  T result{};
  size_t size = range_end - range_start;
  for (size_t i = 0; i < size; ++i)
  {
    result += range_start[i];
  }

  return result;
}



}
#endif
//-------------------------------------------------------------------------
