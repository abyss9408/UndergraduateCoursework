/*!************************************************************************
    \file array.tpp

    \author Bryan Ang Wei Ze

    \par DP email: bryanweize.ang\@digipen.edu

    \par Course: CSD1171 High-Level Programming 2

    \par Lab #8

    \date 03-15-2024
    
    \brief
    This file defines the member and non-member functions that emulate 
    the behaviour of std::array
**************************************************************************/
template <typename T, size_t N>
typename Array<T, N>::iterator Array<T, N>::begin() noexcept
{
    return arraylist;
}

template <typename T, size_t N>
typename Array<T, N>::const_iterator Array<T, N>::begin() const noexcept
{
    return arraylist;
}

template <typename T, size_t N>
typename Array<T, N>::iterator Array<T, N>::end() noexcept
{
    return arraylist+N;
}

template <typename T, size_t N>
typename Array<T, N>::const_iterator Array<T, N>::end() const noexcept
{
    return arraylist+N;
}

template <typename T, size_t N>
typename Array<T, N>::const_iterator Array<T, N>::cbegin() const noexcept
{
    return arraylist;
}

template <typename T, size_t N>
typename Array<T, N>::const_iterator Array<T, N>::cend() const noexcept
{
    return arraylist+N;
}

template <typename T, size_t N>
typename Array<T, N>::reference Array<T, N>::front()
{
    return arraylist[0];
}

template <typename T, size_t N>
typename Array<T, N>::const_reference Array<T, N>::front() const
{
    return arraylist[0];
}

template <typename T, size_t N>
typename Array<T, N>::reference Array<T, N>::back()
{
    return arraylist[N-1];
}

template <typename T, size_t N>
typename Array<T, N>::const_reference Array<T, N>::back() const
{
    return arraylist[N-1];
}

template <typename T, size_t N>
typename Array<T, N>::reference Array<T, N>::operator[](size_type pos)
{
    return arraylist[pos];
}

template <typename T, size_t N>
typename Array<T, N>::const_reference Array<T, N>::operator[](size_type pos) const
{
    return arraylist[pos];
}

template <typename T, size_t N>
constexpr bool Array<T, N>::empty() noexcept
{
    return (N == 0) ? true : false;
}

template <typename T, size_t N>
T* Array<T, N>::data() noexcept
{
    return arraylist;
}

template <typename T, size_t N>
const T* Array<T, N>::data() const noexcept
{
    return arraylist;
}

template <typename T, size_t N>
typename Array<T, N>::size_type Array<T, N>::size() const noexcept
{
    return N;
}

template <typename T, size_t N>
void Array<T, N>::fill(const value_type& val)
{
    for (reference element : arraylist)
    {
        element = val;
    }
}

template <typename T, size_t N>
void Array<T, N>::swap(Array& rhs)
{
    std::swap(arraylist, rhs.arraylist);
}

template <typename T, size_t N>
void swap(Array<T, N>& lhs, Array<T, N>& rhs)
{
    std::swap(lhs, rhs);
}

template <typename T, size_t N>
bool operator==(const Array<T, N>& lhs, const Array<T, N>& rhs)
{
    for (size_t i{}; i < N; ++i)
    {
        if (lhs[i] != rhs[i])
        {
            return false;
        }
    }

    return true;
}

template <typename T, size_t N>
bool operator!=(const Array<T, N>& lhs, const Array<T, N>& rhs)
{
    return !(lhs == rhs);
}

template <typename T, size_t N>
bool operator<(const Array<T, N>& lhs, const Array<T, N>& rhs)
{
    for (size_t i{}; i < N; ++i)
    {
        if (lhs[i] < rhs[i])
        {
            return true;
        }
        else if (lhs[i] > rhs[i])
        {
            return false;
        }
    }

    return false;
}

template <typename T, size_t N>
bool operator>(const Array<T, N>& lhs, const Array<T, N>& rhs)
{
    for (size_t i{}; i < N; ++i)
    {
        if (lhs[i] > rhs[i])
        {
            return true;
        }
        else if (lhs[i] < rhs[i])
        {
            return false;
        }
    }

    return false;
}