/*!************************************************************************
    \file array.hpp

    \author Bryan Ang Wei Ze

    \par DP email: bryanweize.ang\@digipen.edu

    \par Course: CSD1171 High-Level Programming 2

    \par Lab #8

    \date 03-15-2024
    
    \brief
    This header file declares class vector, member and non-member functions
    that emulate the behaviour of std::array
**************************************************************************/
//-------------------------------------------------------------------------
#ifndef ARRAY_HPP_
#define ARRAY_HPP_
//-------------------------------------------------------------------------

#include <cstddef> // for size_t
#include <algorithm>

namespace hlp2 {

template <typename T, size_t N>
struct Array
{

    using size_type = size_t;
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using iterator = pointer;
    using const_iterator = const_pointer;
    using reference = value_type&;
    using const_reference = const value_type&;
    
    /***************************************************************************/
    /*!
    \brief
    Returns an iterator pointing to the first element in the array container

    \return
    An iterator to the beginning of the array
    */
    /**************************************************************************/
    iterator begin() noexcept;

    /***************************************************************************/
    /*!
    \brief
    Returns an iterator pointing to the first element in the array container

    \return
    An iterator to the beginning of the array
    */
    /**************************************************************************/
    const_iterator begin() const noexcept;

    /***************************************************************************/
    /*!
    \brief
    Returns an iterator pointing to the past-the-end element in the array container.

    \return
    An iterator to the element past the end of the array
    */
    /**************************************************************************/
    iterator end() noexcept;

    /***************************************************************************/
    /*!
    \brief
    Returns an iterator pointing to the past-the-end element in the array container.

    \return
    An iterator to the element past the end of the array
    */
    /**************************************************************************/
    const_iterator end() const noexcept;

    /***************************************************************************/
    /*!
    \brief
    Returns a const_iterator pointing to the first element in the array container.

    \return
    A const_iterator to the beginning of the array
    */
    /**************************************************************************/
    const_iterator cbegin() const noexcept;

    /***************************************************************************/
    /*!
    \brief
    Returns a const_iterator pointing to the past-the-end element in the array container.

    \return
    A const_iterator to the element past the end of the array
    */
    /**************************************************************************/
    const_iterator cend() const noexcept;

    /***************************************************************************/
    /*!
    \brief
    Returns a reference to the first element in the array container.

    \return
    A reference to the first element in the array.
    */
    /**************************************************************************/
    reference front();

    /***************************************************************************/
    /*!
    \brief
    Returns a reference to the first element in the array container.

    \return
    A reference to the first element in the array.
    */
    /**************************************************************************/
    const_reference front() const;
    
    /***************************************************************************/
    /*!
    \brief
    Returns a reference to the last element in the array container.

    \return
    A reference to the last element in the array.
    */
    /**************************************************************************/
    reference back();

    /***************************************************************************/
    /*!
    \brief
    Returns a reference to the last element in the array container.

    \return
    A reference to the last element in the array.
    */
    /**************************************************************************/
    const_reference back() const;

    /***************************************************************************/
    /*!
    \brief
    Returns a reference to the element at position pos in the array container.

    \param pos
    Position of an element in the array.

    \return
    The element at the specified position in the array.
    */
    /**************************************************************************/
    reference operator[](size_type pos);

    /***************************************************************************/
    /*!
    \brief
    Returns a reference to the element at position pos in the array container.

    \param pos
    Position of an element in the array.

    \return
    The element at the specified position in the array.
    */
    /**************************************************************************/
    const_reference operator[](size_type pos) const;
    
    /***************************************************************************/
    /*!
    \brief
    Returns a bool value indicating whether the array container is empty.

    \return
    true if the array size is 0, false otherwise.
    */
    /**************************************************************************/
    constexpr bool empty() noexcept;

    /***************************************************************************/
    /*!
    \brief
    Returns a pointer to the first element in the array object.

    \return
    Pointer to the data contained by the array object.
    */
    /**************************************************************************/
    value_type* data() noexcept;
    
    /***************************************************************************/
    /*!
    \brief
    Returns a pointer to the first element in the array object.

    \return
    Pointer to the data contained by the array object.
    */
    /**************************************************************************/
    const value_type* data() const noexcept;
    
    /***************************************************************************/
    /*!
    \brief
    Returns the number of elements in the array container.

    \return
    The number of elements contained in the array object.
    */
    /**************************************************************************/
    size_type size() const noexcept;

    /***************************************************************************/
    /*!
    \brief
    Sets val as the value for all the elements in the array object.

    \param pos
    Value to fill the array with.
    */
    /**************************************************************************/
    void fill(const value_type& val);

    /***************************************************************************/
    /*!
    \brief
    Exchanges the content of the array by the content of rhs, which is another 
    array object of the same type (including the same size).

    \param rhs
    Another array container of the same type
    */
    /**************************************************************************/
    void swap(Array& rhs);

    T arraylist[N];
};

/***************************************************************************/
    /*!
    \brief
    Exchanges the content of lhs by the content of rhs, which both lhs and rhs
    are Array containers of the same type and size

    \param lhs
    First array container of the same type

    \param rhs
    Second array container of the same type
    */
/**************************************************************************/
template <typename T, size_t N>
void swap(Array<T, N>& lhs, Array<T, N>& rhs);

/***************************************************************************/
    /*!
    \brief
    Compares the elements sequentially using operator==, stopping at the first mismatch

    \param lhs
    First array container of the same type

    \param rhs
    Second array container of the same type

    \return 
    true if the corresponding elements of both array containers are the same,
    false otherwise
    */
/**************************************************************************/
template <typename T, size_t N>
bool operator==(const Array<T, N>& lhs, const Array<T, N>& rhs);

/***************************************************************************/
    /*!
    \brief
    Compares the elements sequentially using operator!=, stopping at the first match

    \param lhs
    First array container of the same type

    \param rhs
    Second array container of the same type

    \return 
    true if the corresponding elements of both array containers are different,
    false otherwise
    */
/**************************************************************************/
template <typename T, size_t N>
bool operator!=(const Array<T, N>& lhs, const Array<T, N>& rhs);

/***************************************************************************/
    /*!
    \brief
    Compares the elements sequentially using operator< in a lexicographical manner 
    and stopping at the first occurrence.

    \param lhs
    First array container of the same type

    \param rhs
    Second array container of the same type

    \return 
    true if the first array compares lexicographically less than the second.
    false otherwise.
    */
/**************************************************************************/
template <typename T, size_t N>
bool operator<(const Array<T, N>& lhs, const Array<T, N>& rhs);

/***************************************************************************/
    /*!
    \brief
    Compares the elements sequentially using operator> in a lexicographical manner 
    and stopping at the first occurrence.

    \param lhs
    First array container of the same type

    \param rhs
    Second array container of the same type

    \return 
    true if the first array compares lexicographically more than the second.
    false otherwise.
    */
/**************************************************************************/
template <typename T, size_t N>
bool operator>(const Array<T, N>& lhs, const Array<T, N>& rhs);

#include "array.tpp"

} // end namespace hlp2

#endif // end ARRAY_HPP_
