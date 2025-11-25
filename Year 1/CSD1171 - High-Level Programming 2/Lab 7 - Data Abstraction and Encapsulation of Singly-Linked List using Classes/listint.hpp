/*!************************************************************************
    \file listint.hpp

    \author Bryan Ang Wei Ze

    \par DP email: bryanweize.ang\@digipen.edu

    \par Course: CSD1171 High-Level Programming 2

    \par Lab #7

    \date 08-03-2024
    
    \brief
    This header file defines a class named ListInt, declares constructors,
    member function operator overloads and non-member function operator
    overloads for the class ListInt
**************************************************************************/

#ifndef LISTINT_HPP
#define LISTINT_HPP

#include <cstddef>
#include <iostream>
#include <iomanip>
#include <algorithm>

namespace hlp2
{
    class ListInt
    {
    public:
        using size_type = size_t;
        using value_type = int;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using reference = value_type&;
        using const_reference = const value_type&;
        // Declare this as a friend function in class ListInt:
        friend std::ostream& operator<<(std::ostream& os, const ListInt& rhs);

    public:
        /***************************************************************************/
        /*!
        \brief
        Default Constructor
        */
        /**************************************************************************/
        ListInt();

        /***************************************************************************/
        /*!
        \brief
        Copy Constructor
        */
        /**************************************************************************/
        ListInt(const ListInt& rhs);

        /***************************************************************************/
        /*!
        \brief
        Single-argument Constructor that takes in std::initializer_list<value_type>

        \param rhs
        std::initializer_list of value_type
        */
        /**************************************************************************/
        ListInt(const std::initializer_list<value_type>& rhs);

        /***************************************************************************/
        /*!
        \brief
        Destructor
        */
        /**************************************************************************/
        ~ListInt();

        /***************************************************************************/
        /*!
        \brief
        Copy assignment with another ListInt object

        \param rhs
        Another ListInt object

        \return
        Reference to invoking ListInt object
        */
        /**************************************************************************/
        ListInt& operator=(const ListInt& rhs);

        /***************************************************************************/
        /*!
        \brief
        Copy assignment with std::initializer_list<value_type>

        \param rhs
        std::initializer_list of value_type

        \return
        Reference to invoking ListInt object
        */
        /**************************************************************************/
        ListInt& operator=(const std::initializer_list<value_type>& rhs);

        /***************************************************************************/
        /*!
        \brief
        Concatenates with nodes of another ListInt container

        \param rhs
        Another ListInt object

        \return
        Reference to invoking ListInt object
        */
        /**************************************************************************/
        ListInt& operator+=(const ListInt& rhs);

        /***************************************************************************/
        /*!
        \brief
        Concatenates with std::initializer_list<value_type>

        \param rhs
        std::initializer_list of value_type

        \return
        Reference to invoking ListInt object
        */
        /**************************************************************************/
        ListInt& operator+=(const std::initializer_list<value_type>& rhs);

        /***************************************************************************/
        /*!
        \brief
        Returns int data of node at position index in ListInt object

        \param index
        Position in ListInt object

        \return
        reference to data of node at position index in ListInt object
        */
        /**************************************************************************/
        reference operator[](size_type index);

        /***************************************************************************/
        /*!
        \brief
        Returns int data of node at position index in ListInt object

        \param index
        Position in ListInt object

        \return
        const reference to data of node at position index in ListInt object
        */
        /**************************************************************************/
        const_reference operator[](size_type index) const;

        /***************************************************************************/
        /*!
        \brief
        Adds value_type to front of the list

        \param value
        Value to be added
        */
        /**************************************************************************/
        void push_front(value_type value);

        /***************************************************************************/
        /*!
        \brief
        Uses tail data member to efficiently add value_type to back of the list

        \param value
        Value to be added
        */
        /**************************************************************************/
        void push_back(value_type value);

        /***************************************************************************/
        /*!
        \brief
        Returns the value of node at front of the list and then destroys this front node

        \return
        Value of node at front of the list
        */
        /**************************************************************************/
        value_type pop_front();

        /***************************************************************************/
        /*!
        \brief
        Rrases all nodes in linked list
        */
        /**************************************************************************/
        void clear();

        /***************************************************************************/
        /*!
        \brief
        Exchanges contents of container with another ListInt container

        \param rhs
        Another ListInt object
        */
        /**************************************************************************/
        void swap(ListInt& rhs);

        /***************************************************************************/
        /*!
        \brief
        Returns number of nodes

        \return
        The number of nodes in a ListInt container
        */
        /**************************************************************************/
        size_type size() const;

        /***************************************************************************/
        /*!
        \brief
        Returns true if linked list is empty; otherwise false

        \return
        true if linked list is empty; otherwise false
        */
        /**************************************************************************/
        bool empty() const;  

        /***************************************************************************/
        /*!
        \brief
        Returns the number of ListInt objects active

        \return
        The number of ListInt objects active
        */
        /**************************************************************************/
        static size_type object_count();

    private:
        struct Node
        {
            value_type data;
            Node *next;
        };
        Node *head{nullptr};
        Node *tail{nullptr};
        size_type counter{0};
        static size_type object_counter;

        /***************************************************************************/
        /*!
        \brief
        Dynamically allocate an object of Node with data

        \return
        A pointer to the dynamically allocated object
        */
        /**************************************************************************/
        Node* new_node(value_type data) const;
    };

    /***************************************************************************/
    /*!
    \brief
    Add two ListInt objects

    \param lhs
    First ListInt object

    \param rhs
    Second ListInt object

    \return
    Result of adding two ListInt objects
    */
    /**************************************************************************/
    ListInt operator+(const ListInt &lhs, const ListInt &rhs);

    /***************************************************************************/
    /*!
    \brief
    Add ListInt object and std::initializer_list<ListInt::value_type>

    \param lhs
    ListInt object

    \param rhs
    std::initializer_list of value_type

    \return
    Result of adding ListInt object and std::initializer_list<ListInt::value_type>
    */
    /**************************************************************************/
    ListInt operator+(const ListInt &lhs, const std::initializer_list<ListInt::value_type> &rhs);

    /***************************************************************************/
    /*!
    \brief
    Add std::initializer_list<ListInt::value_type> and ListInt object

    \param lhs
    std::initializer_list of value_type

    \param rhs
    ListInt object

    \return
    Result of std::initializer_list<ListInt::value_type> and ListInt object
    */
    /**************************************************************************/
    ListInt operator+(const std::initializer_list<ListInt::value_type> &lhs, const ListInt &rhs);

}
#endif
