/*!************************************************************************
    \file vector.cpp

    \author Bryan Ang Wei Ze

    \par DP email: bryanweize.ang\@digipen.edu

    \par Course: CSD1171 High-Level Programming 2

    \par Programming Assignment #6

    \date 02-28-2024
    
    \brief
    This source file implements member functions that emulate the behaviour of
    std::vector
**************************************************************************/
#include "vector.hpp"

namespace hlp2
{
    vector::vector() 
    : sz{0}, space{0}, allocs{0}, data{nullptr}
    {
        
    }

    vector::vector(size_type n) 
    : sz{n}, space{n}, allocs{1}, data{new value_type[sz]}
    {
        for (size_t i{}; i < sz; ++i)
        {
            data[i] = 0;
        }
    }

    vector::vector(std::initializer_list<int> rhs) 
    : sz{rhs.size()}, space{rhs.size()}, allocs{1}, data{new value_type[sz]}
    {
        pointer tmp{data};
        for (int val : rhs)
        {
            *tmp++ = val;
        }
    }

    vector::vector(const vector& rhs) 
    : sz{rhs.sz}, space{rhs.sz}, allocs{1}, data{new value_type[sz]}
    {
        for (size_t i{}; i < sz; ++i)
        {
            data[i] = rhs.data[i];
        }
    }

    vector::~vector()
    {
        delete[] data;
    }

    vector& vector::operator=(const vector &rhs)
    {
        sz = rhs.sz;
        space = rhs.sz;
        allocs += 1;
        value_type *p = new value_type[sz];
        for (size_type i{}; i < sz; ++i)
        {
            p[i] = rhs.data[i];
        }
        delete[] data;
        data = p;
        return *this;
    }

    vector& vector::operator=(std::initializer_list<int> rhs)
    {
        sz = rhs.size();
        space = rhs.size();
        allocs += 1;
        value_type *p = new value_type[sz];
        for (size_type i{}; i < sz; ++i)
        {
            p[i] = *(rhs.begin() + i);
        }
        delete[] data;
        data = p;
        return *this;
    }

    vector::reference vector::operator[](size_type index)
    {
        return data[index];
    }

    vector::const_reference vector::operator[](size_type index) const
    {
        return data[index];
    }

    vector::pointer vector::begin()
    {
        return data;
    }

    vector::pointer vector::end()
    {
        return data+sz;
    }

    vector::const_pointer vector::begin() const
    {
        return data;
    }

    vector::const_pointer vector::end() const
    {
        return data+sz;
    }

    vector::const_pointer vector::cbegin() const
    {
        return data;
    }

    vector::const_pointer vector::cend() const
    {
        return data+sz;
    }

    void vector::reserve(size_type newsize)
    {
        if (newsize <= space)
        {
            return;
        }

        pointer ptr = new value_type[newsize]{0};
        for (size_type i{}; i < sz; ++i)
        {
            ptr[i] = data[i];
        }
        delete[] data;
        data = ptr;
        space = newsize;
        ++allocs;
    }

    void vector::resize(size_type newsize)
    {
        if (newsize > space)
        {
            reserve(newsize);
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

    void vector::push_back(value_type value)
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
    
    bool vector::empty() const
    {
        if (sz == 0)
        {
            return true;
        }
        
        return false;
    }
    vector::size_type vector::size() const
    {
        return sz;
    }
    vector::size_type vector::capacity() const
    {
        return space;
    }
    vector::size_type vector::allocations() const
    {
        return allocs;
    }
}