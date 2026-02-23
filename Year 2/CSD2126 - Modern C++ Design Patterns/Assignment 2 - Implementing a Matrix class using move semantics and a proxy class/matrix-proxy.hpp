/*!************************************************************************
  \file matrix-proxy.hpp

  \author Bryan Ang Wei Ze

  \par DP email: bryanweize.ang\@digipen.edu

  \par Course: CSD2126 Modern C++ Design Patterns

  \par Programming Assignment #2

  \date 25-09-2024
  
  \brief
  This header file containing definition of class template Matrix,
  declarations of non-member functions, definitions of member functions
  outside definition class Matrix, and definitions of non-member functions.
**************************************************************************/

// First, define the class template. Then, define the member functions of the class template
// OUTSIDE the class definition. The automatic grader will not be able to check for this - 
// however, the manual grader can. Occurrences of EACH such function defined in the class definition
//  will result in a deduction of ONE LETTER GRADE!!! You're forewarned!!!

// This is how you should implement the assignment:

#ifndef MATRIX_PROXY_HPP
#define MATRIX_PROXY_HPP

// include necessary headers ...
#include <initializer_list>
#include <algorithm>

namespace HLP3
{
    template <typename T>
    class Matrix
    {
    public:
        // provide common standard library container type definitions
        using size_type = size_t;
        using value_type = T;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using reference = value_type&;
        using const_reference = const value_type&;

        class RowProxy
        {
        public:
            RowProxy(Matrix& matrix, size_type row);
            reference operator[](size_type col);
        private:
            Matrix& matrix_;
            size_type row_;
        };

        class ConstRowProxy
        {
        public:
            ConstRowProxy(const Matrix& matrix, size_type row);
            const_reference operator[](size_type col) const;
        private:
            const Matrix& matrix_;
            size_type row_;
        };

    public:
        /***************************************************************************/
        /*!
        \brief
        Constructor that creates nr x nc matrix

        \param nr
        Number of rows

        \param nc
        Number of columns

        */
        /**************************************************************************/
        Matrix(size_type nr, size_type nc);

        /***************************************************************************/
        /*!
        \brief
        Copy constructor that creates a matrix that is a deep copy of matrix rhs

        \param rhs
        Matrix to copy construct from
        */
        /**************************************************************************/
        Matrix(Matrix const& rhs);

        /***************************************************************************/
        /*!
        \brief
        Move constructor that move constructs a matrix from matrix rhs

        \param rhs
        Matrix to move construct from
        */
        /**************************************************************************/
        Matrix(Matrix&& rhs) noexcept;

        /***************************************************************************/
        /*!
        \brief
        Constructor that creates nr x nc matrix from an initializer list that has
        nr rows and nc columns

        \param list
        Initializer list to construct matrix from
        */
        /**************************************************************************/
        Matrix(std::initializer_list<std::initializer_list<value_type>> list);

        /***************************************************************************/
        /*!
        \brief
        Destructor that destroys the matrix by explicitly returning storage to free store.
        */
        /**************************************************************************/
        ~Matrix() noexcept;

        /***************************************************************************/
        /*!
        \brief
        Copy assignment operator that replaces the matrix with a deep copy of rhs

        \param rhs
        Matrix to deep copy from

        \return
        New matrix
        */
        /**************************************************************************/
        Matrix& operator=(Matrix const& rhs);

        /***************************************************************************/
        /*!
        \brief
        Move assignment operator that replaces the matrix with a move of rhs

        \param rhs
        Matrix to move from

        \return
        New matrix
        */
        /**************************************************************************/
        Matrix& operator=(Matrix&& rhs) noexcept;
        
        /***************************************************************************/
        /*!
        \brief
        Returns the number of rows in matrix

        \return
        Number of rows
        */
        /**************************************************************************/
        size_type get_rows() const noexcept;

        /***************************************************************************/
        /*!
        \brief
        Returns the number of columns in matrix

        \return
        Number of columns
        */
        /**************************************************************************/
        size_type get_cols() const noexcept;
        
        /***************************************************************************/
        /*!
        \brief
        Subscipting operator that returns index where rth row of matrix data

        \param r
        Row index

        \return
        An object of nested RowProxy class
        */
        /**************************************************************************/
        RowProxy operator[](size_type r);

        /***************************************************************************/
        /*!
        \brief
        Subscipting operator that returns index where rth row of matrix data

        \param r
        Row index

        \return
        An object of nested ConstRowProxy class
        */
        /**************************************************************************/
        ConstRowProxy operator[](size_type r) const;

    private:
        size_type rows;
        size_type cols;
        pointer data;
    };
    
    // RowProxy member functions
    template <typename T>
    Matrix<T>::RowProxy::RowProxy(Matrix& matrix, size_type row)
        : matrix_(matrix), row_(row) {}

    template <typename T>
    typename Matrix<T>::reference Matrix<T>::RowProxy::operator[](size_type col)
    {
        if (col >= matrix_.cols)
        {
            throw std::out_of_range("Column index out of range");
        }
        return matrix_.data[row_ * matrix_.cols + col];
    }

    // ConstRowProxy member functions
    template <typename T>
    Matrix<T>::ConstRowProxy::ConstRowProxy(const Matrix& matrix, size_type row)
        : matrix_(matrix), row_(row) {}

    template <typename T>
    typename Matrix<T>::const_reference Matrix<T>::ConstRowProxy::operator[](size_type col) const
    {
        if (col >= matrix_.cols)
        {
            throw std::out_of_range("Column index out of range");
        }
        return matrix_.data[row_ * matrix_.cols + col];
    }

    // Matrix member functions
    template <typename T>
    Matrix<T>::Matrix(size_type nr, size_type nc)
    : rows(nr), cols(nc), data(new value_type[rows * cols]) {}

    template <typename T>
    Matrix<T>::Matrix(Matrix<T> const& rhs)
    : rows(rhs.rows), cols(rhs.cols), data(new value_type[rows * cols])
    {
        std::copy(rhs.data, rhs.data + (rows * cols), data);
    }

    template <typename T>
    Matrix<T>::Matrix(Matrix<T>&& rhs) noexcept
    : rows(rhs.rows), cols(rhs.cols), data(rhs.data)
    {
        rhs.rows = 0;
        rhs.cols = 0;
        rhs.data = nullptr;
    }

    template <typename T>
    Matrix<T>::Matrix(std::initializer_list<std::initializer_list<value_type>> list)
    : rows(list.size()), cols(list.begin()->size()), data(new value_type[rows * cols])
    {
        size_type r = 0;
        for (const auto& row : list)
        {
            if (row.size() != cols)
            {
                delete[] data;
                throw std::runtime_error("bad initializer list");
            }
            std::copy(row.begin(), row.end(), data + r * cols);
            ++r;
        }
    }

    template <typename T>
    Matrix<T>::~Matrix() noexcept
    {
        delete[] data;
    }

    template <typename T>
    typename Matrix<T>::Matrix& Matrix<T>::operator=(Matrix<T> const& rhs)
    {
        Matrix tmp(rhs);
        std::swap(rows, tmp.rows);
        std::swap(cols, tmp.cols);
        std::swap(data, tmp.data);

        return *this;
    }

    template <typename T>
    typename Matrix<T>::Matrix& Matrix<T>::operator=(Matrix<T>&& rhs) noexcept
    {
        delete[] data;
        rows = rhs.rows;
        cols = rhs.cols;
        data = rhs.data;
        rhs.rows = 0;
        rhs.cols = 0;
        rhs.data = nullptr;

        return *this;
    }

    template <typename T>
    typename Matrix<T>::size_type Matrix<T>::get_rows() const noexcept
    {
        return rows;
    }

    template <typename T>
    typename Matrix<T>::size_type Matrix<T>::get_cols() const noexcept
    {
        return cols;
    }

    template <typename T>
    typename Matrix<T>::RowProxy Matrix<T>::operator[](size_type r)
    {
        if (r >= rows)
        {
            throw std::out_of_range("Row index out of range");
        }
        
        return RowProxy(*this, r);
    }

    template <typename T>
    typename Matrix<T>::ConstRowProxy Matrix<T>::operator[](size_type r) const
    {
        if (r >= rows)
        {
            throw std::out_of_range("Row index out of range");
        }
        
        return ConstRowProxy(*this, r);
    }

    // Global functions of Matrix API
    template <typename T>
    Matrix<T> operator+(const Matrix<T>& M, const Matrix<T>& N)
    {
        if (M.get_rows() != N.get_rows() || M.get_cols() != N.get_cols())
        {
            throw std::runtime_error("operands for matrix addition must have same dimensions");
        }

        Matrix<T> result(M.get_rows(), M.get_cols());
        for (typename Matrix<T>::size_type i = 0; i < M.get_rows(); ++i)
        {
            for (typename Matrix<T>::size_type j = 0; j < M.get_cols(); ++j)
            {
                result[i][j] = M[i][j] + N[i][j];
            }
        }
        return result;
    }

    template <typename T>
    Matrix<T> operator-(const Matrix<T>& M, const Matrix<T>& N)
    {
        if (M.get_rows() != N.get_rows() || M.get_cols() != N.get_cols())
        {
            throw std::runtime_error("operands for matrix subtraction must have same dimensions");
        }

        Matrix<T> result(M.get_rows(), M.get_cols());
        for (typename Matrix<T>::size_type i = 0; i < M.get_rows(); ++i)
        {
            for (typename Matrix<T>::size_type j = 0; j < M.get_cols(); ++j)
            {
                result[i][j] = M[i][j] - N[i][j];
            }
        }
        return result;
    }

    template <typename T>
    Matrix<T> operator*(const Matrix<T>& M, const Matrix<T>& N)
    {
        if (M.get_cols() != N.get_rows())
        {
            throw std::runtime_error("number of columns in left operand must match number of rows in right operand");
        }

        Matrix<T> result(M.get_rows(), N.get_cols());
        for (typename Matrix<T>::size_type i = 0; i < M.get_rows(); ++i)
        {
            for (typename Matrix<T>::size_type j = 0; j < N.get_cols(); ++j)
            {
                T sum = T();
                for (typename Matrix<T>::size_type k = 0; k < M.get_cols(); ++k)
                {
                    sum += M[i][k] * N[k][j];
                }
                result[i][j] = sum;
            }
        }
        return result;
    }

    template <typename T>
    Matrix<T> operator*(const T& r, const Matrix<T>& M)
    {
        Matrix<T> result(M.get_rows(), M.get_cols());
        for (typename Matrix<T>::size_type i = 0; i < M.get_rows(); ++i)
        {
            for (typename Matrix<T>::size_type j = 0; j < M.get_cols(); ++j)
            {
                result[i][j] = r * M[i][j];
            }
        }
        return result;
    }

    template <typename T>
    bool operator==(const Matrix<T>& M, const Matrix<T>& N)
    {
        if (M.get_rows() != N.get_rows() || M.get_cols() != N.get_cols())
        {
            return false;
        }

        for (typename Matrix<T>::size_type i = 0; i < M.get_rows(); ++i)
        {
            for (typename Matrix<T>::size_type j = 0; j < M.get_cols(); ++j)
            {
                if (M[i][j] != N[i][j])
                {
                    return false;
                }
            }
        }
        return true;
    }

    template <typename T>
    bool operator!=(const Matrix<T>& M, const Matrix<T>& N)
    {
        return !(M == N);
    }
}
#endif