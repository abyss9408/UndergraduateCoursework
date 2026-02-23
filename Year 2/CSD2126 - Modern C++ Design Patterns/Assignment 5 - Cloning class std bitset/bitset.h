/*!************************************************************************
  \file bitset.h

  \author Bryan Ang Wei Ze

  \par DP email: bryanweize.ang\@digipen.edu

  \par Course: CSD2126 Modern C++ Design Patterns

  \par Programming Assignment #5

  \date 31-10-2024
  
  \brief
  This header file contains definition of class template bitset and declarations
  of member functions
**************************************************************************/
#ifndef BITSET_H
#define BITSET_H

namespace HLP3
{
    template <size_t N>
    class bitset
    {
    public:

        /***************************************************************************/
            /*!
            \brief
            Default constructor that allocates memory
            */
        /**************************************************************************/
        bitset();

        /***************************************************************************/
            /*!
            \brief
            Copy constuctor

            \param rhs
            Another bitset with same number of bits
            */
        /**************************************************************************/
        bitset(const bitset& rhs);

        /***************************************************************************/
            /*!
            \brief
            Copy assignment

            \param rhs
            Another bitset with same number of bits
            */
        /**************************************************************************/
        bitset& operator=(const bitset& rhs);

        /***************************************************************************/
            /*!
            \brief
            Move constuctor

            \param rhs
            Another bitset with same number of bits
            */
        /**************************************************************************/
        bitset(bitset &&rhs) noexcept;

        /***************************************************************************/
            /*!
            \brief
            Move assignment

            \param rhs
            Another bitset with same number of bits
            */
        /**************************************************************************/
        bitset& operator=(bitset &&rhs) noexcept;

        /***************************************************************************/
            /*!
            \brief
            Destructor that deallocates memory
            */
        /**************************************************************************/
        ~bitset();

        /***************************************************************************/
            /*!
            \brief
            Sets bits.

            \param pos
            Order position of the bit whose value is modified.
            Order positions are counted from the rightmost bit, which is order position 0.
            If pos is equal or greater than the bitset size, an out_of_range exception is thrown.

            \param val
            Value to store in the bit (either true for one or false for zero)
            */
        /**************************************************************************/
        void set(size_t pos, bool val = true);

        /***************************************************************************/
            /*!
            \brief
            Resets bits to zero.

            \param pos
            Order position of the bit whose value is modified.
            Order positions are counted from the rightmost bit, which is order position 0.
            If pos is equal or greater than the bitset size, an out_of_range exception is thrown.
            */
        /**************************************************************************/
        void reset(size_t pos);

        /***************************************************************************/
            /*!
            \brief
            Flips bit values converting zeros into ones and ones into zeros.

            \param pos
            Order position of the bit whose value is filped.
            Order positions are counted from the rightmost bit, which is order position 0.
            If pos is equal or greater than the bitset size, an out_of_range exception is thrown.
            */
        /**************************************************************************/
        void flip(size_t pos);

        /***************************************************************************/
            /*!
            \brief
            Constructs a std::string object that represents the bits in the bitset as a
            succession of zeros and/or ones.

            \param zero
            Character value to represent zero

            \param one
            Character value to represent one

            \return
            A string representing the bits in the bitset.
            */
        /**************************************************************************/
        std::string to_string(char zero = '0', char one = '1') const;

        /***************************************************************************/
            /*!
            \brief
            Returns whether the bit at position pos is set.

            \param pos
            Order position of the bit whose value is retrieved.
            Order positions are counted from the rightmost bit, which is order position 0.

            \return
            true if the bit at position pos is set, and false if it is not set.
            */
        /**************************************************************************/
        bool test(size_t pos) const;

        /***************************************************************************/
            /*!
            \brief
            Returns the number of bits in the bitset.

            \return
            The number of bits in the bitset.
            */
        /**************************************************************************/
        constexpr size_t size() noexcept;

        /***************************************************************************/
            /*!
            \brief
            The function returns the value (or a reference) to the bit at position pos.

            \param pos
            Order position of the bit whose value is accessed.
            Order positions are counted from the rightmost bit, which is order position 0.

            \return
            The bit at position pos.
            */
        /**************************************************************************/
        bool operator[] (size_t pos) const;

        /***************************************************************************/
            /*!
            \brief
            Returns the number of bits in the bitset that are set.

            \return
            The number of bits set.
            */
        /**************************************************************************/
        size_t count() const noexcept;
    private:
        bool* data;
    };
}

#include "bitset.hpp"

#endif