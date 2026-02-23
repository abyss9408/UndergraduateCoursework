/*!************************************************************************
  \file polynomial.h

  \author Bryan Ang Wei Ze

  \par DP email: bryanweize.ang\@digipen.edu

  \par Course: CSD2126 Modern C++ Design Patterns

  \par Programming Quiz #1

  \date 17-09-2024
  
  \brief
  This header file declares a class template called Polynomial which
  emulates the mathematical expression
**************************************************************************/

#ifndef POLYNOMIAL_H
#define POLYNOMIAL_H

#include <iostream> // std::ostream

namespace HLP3 {

// declare class template Polynomial
template <typename T, int N>
class Polynomial
{
public:
    
    /***************************************************************************/
    /*!
    \brief
    Default constructor that creates a zero polynomial
    */
    /**************************************************************************/
    Polynomial();
    
    /***************************************************************************/
    /*!
    \brief
    Single argument constructor that creates a polynomial from another polynomial,
    which has the same degree but coefficient types could be different.
    
    \param rhs
    Polynomial to be constructed from
    */
    /**************************************************************************/
    template <typename U>
    Polynomial(const Polynomial<U, N>& rhs);
    
    /***************************************************************************/
    /*!
    \brief
    Copy assignment that assigns a Polynomial to another Polynomial,
    which has the same degree but coefficient types could be different.
    
    \param rhs
    Polynomial to be assigned from
    */
    /**************************************************************************/
    template <typename U>
    Polynomial& operator=(const Polynomial<U, N>& rhs);
    
    /***************************************************************************/
    /*!
    \brief
    Subscript operator overload that allows writing of coefficient values
    
    \param index
    
    \return T
    reference to coefficient
    Polynomial to be assigned from
    */
    /**************************************************************************/
    T& operator[](int index);
    
    /***************************************************************************/
    /*!
    \brief
    Const subscript operator overload that allows reading of coefficient values
    
    \param index
    
    \return T
    const reference to coefficient
    Polynomial to be assigned from
    */
    /**************************************************************************/
    const T& operator[](int index) const;
    
    /***************************************************************************/
    /*!
    \brief
    Multiplication operator overload that multiplications of Polynomial of different
    degrees, but must be the same coefficient type.
    
    \param rhs
    Other Polynomial to multiply with
    
    \return T
    Result of multiplication
    */
    /**************************************************************************/
    template <int M>
    Polynomial<T, N + M> operator*(const Polynomial<T, M>& rhs);
    
    /***************************************************************************/
    /*!
    \brief
    Function call operator overload that evaluates the Polynomial using argument
    val_x.
    
    \param val_x
    Argument to evaluate Polynomial
    
    \return T
    Result of evaluation
    */
    /**************************************************************************/
    T operator()(T val_x);
private:
    T data[N + 1];
};


}

#include "polynomial.tpp"

#endif
