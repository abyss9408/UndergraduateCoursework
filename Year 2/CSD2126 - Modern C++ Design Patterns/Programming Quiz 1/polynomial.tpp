/*!************************************************************************
  \file polynomial.tpp

  \author Bryan Ang Wei Ze

  \par DP email: bryanweize.ang\@digipen.edu

  \par Course: CSD2126 Modern C++ Design Patterns

  \par Programming Quiz #1

  \date 17-09-2024
  
  \brief
  This source file defines the member functions of the class template
  Polynomial
**************************************************************************/

namespace HLP3 {

// Define member functions of class template Polynomial ...
template <typename T, int N> 
Polynomial<T, N>::Polynomial()
{
  for (size_t i = 0; i < N + 1; ++i)
  {
    data[i] = static_cast<T>(0);
  }
}

template <typename T, int N>
template <typename U>
Polynomial<T, N>::Polynomial(const Polynomial<U, N>& rhs)
{
  for (size_t i = 0; i < N + 1; ++i)
  {
    data[i] = rhs[i];
  }
}

template <typename T, int N>
template <typename U>
Polynomial<T, N>& Polynomial<T, N>::operator=(const Polynomial<U, N>& rhs)
{
  for (size_t i = 0; i < N + 1; ++i)
  {
    data[i] = rhs[i];
  }

  return *this;
}

template <typename T, int N>
T& Polynomial<T, N>::operator[](int index)
{
  return data[index];
}

template <typename T, int N>
const T& Polynomial<T, N>::operator[](int index) const
{
  return data[index];
}

template <typename T, int N>
template <int M>
Polynomial<T, N + M> Polynomial<T, N>::operator*(const Polynomial<T, M>& rhs)
{
  Polynomial<T, N + M> tmp;

  for (int i = 0; i < N + 1; ++i)
  {
    for (int j = 0; j < M + 1; ++j)
    {
      tmp[i + j] += data[i] * rhs[j];
    }
  }
  
  return tmp;
}

template <typename T, int N>
T Polynomial<T, N>::operator()(T val_x)
{
  T result = T();
  for (int i = N; i >= 0; --i)
  {
    result = result * val_x + data[i];
  }
  
  return result;
}


// DON'T CHANGE/EDIT THE FOLLOWING DEFINITION:
template< typename T, int N > 
std::ostream& operator<<(std::ostream &out, Polynomial<T, N> const& pol) {
  out << pol[0] << " ";
  for ( int i=1; i<=N; ++i ) {
    if ( pol[i] != 0 ) { // skip terms with zero coefficients
      if      ( pol[i] > 0 ) {  out << "+"; }
      if      ( pol[i] == 1 )  { }
      else if ( pol[i] == -1 ) { out << "-"; }
      else                     { out << pol[i] << "*"; }
      out << "x^" << i << " ";
    }
  }
  return out;
}

} // end namespace HLP3
