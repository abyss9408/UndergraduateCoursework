/*!************************************************************************
    \file test.hpp

    \author Bryan Ang Wei Ze

    \par DP email: bryanweize.ang\@digipen.edu

    \par Course: CSD1171 High-Level Programming 2

    \par Lab #9

    \date 03-23-2024
    
    \brief
    This header file defines five function templates that test divide1, divide2,
    divide3, divide4 and divide5 and throw user-defined division_by_zero exceptions
    when division by zero is encountered. And also, an input stream_wrapper class
    that throws user-defined invalid_input exception for invalid input.
**************************************************************************/

#ifndef TEST_HPP
#define TEST_HPP

#include <limits>
#include <iostream>
#include <string>
#include <exception>
#include <utility>

namespace hlp2
{
    /***************************************************************************/
    /*!
    \brief
    This function template tests the divide1 function that returns an error flag
    and writes the calculated value in an "output-only" reference parameter. 
    It throws an exception when return value of divide1 function is false.
    Otherwise, it prints the result of the division.

    \param numerator
    Numerator to be passed into the divide1 function through template type
    parameter func

    \param denominator
    Denominator to be passed into the divide1 function through template type
    parameter func
    */
    /**************************************************************************/
    template <typename F>
    void test1(int numerator, int denominator, F func);
    
    /***************************************************************************/
    /*!
    \brief
    This function template tests the divide2 function that returns an object that
    aggregates the calculated value and an error flag. It throws an exception when
    the first member of returned object of divide2 function is false. Otherwise,
    it prints the value of the second member.

    \param numerator
    Numerator to be passed into the divide2 function through template type
    parameter func

    \param denominator
    Denominator to be passed into the divide2 function through template type
    parameter func
    */
    /**************************************************************************/
    template <typename F>
    void test2(int numerator, int denominator, F func);
    
    /***************************************************************************/
    /*!
    \brief
    This function template tests the divide3 function that returns the calculated
    value and uses global variable errno to indicate whether the result is valid.
    It throws an exception when errno is set to 0. Otherwise, it prints the result
    of the division.

    \param numerator
    Numerator to be passed into the divide3 function through template type
    parameter func

    \param denominator
    Denominator to be passed into the divide3 function through template type
    parameter func
    */
    /**************************************************************************/
    template <typename F>
    void test3(int numerator, int denominator, F func);
    
    /***************************************************************************/
    /*!
    \brief
    This function template tests the divide4 function that returns the calculated
    value but reserves a specific value to indicate whether the result is valid.
    It throws an exception when the result is smallest int value on machine.
    Otherwise, it prints the result of the division.

    \param numerator
    Numerator to be passed into the divide4 function through template type
    parameter func

    \param denominator
    Denominator to be passed into the divide4 function through template type
    parameter func
    */
    /**************************************************************************/
    template <typename F>
    void test4(int numerator, int denominator, F func);
    
    /***************************************************************************/
    /*!
    \brief
    This function template tests the divide5 function that returns the calculated
    value and indicates an error by throwing an exception. It catches and rethrows
    the exception thrown by divide5 when the denominator is zero. Otherwise, it
    prints the result of the division.

    \param numerator
    Numerator to be passed into the divide5 function through template type
    parameter func

    \param denominator
    Denominator to be passed into the divide5 function through template type
    parameter func
    */
    /**************************************************************************/
    template <typename F>
    void test5(int numerator, int denominator, F func);
    
    // exception class for division by zero error
    class division_by_zero : public std::exception
    {
    public:
        division_by_zero(int numerator)
        {
            reason = "Division by zero: " + std::to_string(numerator) + " / 0!\n";
        }
        const char* what() const throw()
        {
            return reason.c_str();
        }
    private:
        std::string reason;
    };
    
    // exception class for invalid input
    class invalid_input : public std::exception
    {
        std::string reason;
    public:
        invalid_input() : reason{"Invalid input!"}{}
        const char* what() const throw()
        {
            return reason.c_str();
        }
    };
    
    /* input stream wrapper that throws an exception when the input stream enters
    a failed state*/
    class stream_wrapper
    {
    private:
        std::istream &input_stream;

    public:
        stream_wrapper(std::istream &is)
        : input_stream{is}{}
        template <typename T>
        stream_wrapper& operator>>(T& val)
        {
            input_stream >> val;
            if (input_stream.fail())
            {
                throw invalid_input();
            }
        
            return *this;
        }
    
    };

    template <typename F>
    void test1(int numerator, int denominator, F func)
    {
        int value{};
        bool result = func(numerator, denominator, value);

        if(!result)
        {
            std::cout << "Calling function #1.\n";
            throw division_by_zero(numerator);
        }

        std::cout << "Calling function #1; result: "<<value<<".\n";
    }

    template <typename F>
    void test2(int numerator, int denominator, F func)
    {
        std::pair<bool, int> result = func(numerator, denominator);
        if(!result.first)
        {
            std::cout << "Calling function #2.\n";
            throw division_by_zero(numerator);
        }

        std::cout << "Calling function #2; result: "<<result.second<<".\n";
    }

    template <typename F>
    void test3(int numerator, int denominator, F func)
    {
        int result = func(numerator, denominator);
        if(!errno)
        {
            std::cout << "Calling function #3.\n";
            throw division_by_zero(numerator);
        }

        std::cout << "Calling function #3; result: "<<result<<".\n";
    }

    template <typename F>
    void test4(int numerator, int denominator, F func)
    {
        int result = func(numerator, denominator);
        if(result == std::numeric_limits<int>::min())
        {
            std::cout << "Calling function #4.\n";
            throw division_by_zero(numerator);
        }

        std::cout << "Calling function #4; result: "<<result<<".\n";
    }

    template <typename F>
    void test5(int numerator, int denominator, F func)
    {
        int result{};
        try
        {   
            result = func(numerator, denominator);
        }
        catch (std::exception const&)
        {
            std::cout << "Calling function #5.\n";
            throw;
        }

        std::cout << "Calling function #5; result: "<<result<<".\n";
    }
}

#endif