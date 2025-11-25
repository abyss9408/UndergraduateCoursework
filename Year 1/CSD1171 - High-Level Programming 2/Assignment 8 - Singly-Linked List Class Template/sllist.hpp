/*!************************************************************************
    \file sllist.hpp

    \author Bryan Ang Wei Ze

    \par DP email: bryanweize.ang\@digipen.edu

    \par Course: CSD1171 High-Level Programming 2

    \par Programming Assignment #8

    \date 14-03-2024
    
    \brief
    This header file contains declarations of ALL static data, static member,
    class template member and non-member functions of sllist<T>.
**************************************************************************/

#ifndef SLLIST_HPP
#define SLLIST_HPP

#include <cstddef>
#include <iostream>
#include <iomanip>
#include <algorithm>

namespace hlp2
{
    template <typename T>
    class sllist
    {
    public:
        using size_type = size_t;
        using value_type = T;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using reference = value_type&;
        using const_reference = const value_type&;

    public:
        /***************************************************************************/
        /*!
        \brief
        Default Constructor
        */
        /**************************************************************************/
        sllist();

        /***************************************************************************/
        /*!
        \brief
        Copy Constructor
        */
        /**************************************************************************/
        sllist(const sllist& rhs);

        /***************************************************************************/
        /*!
        \brief
        Single-argument Constructor that takes in std::initializer_list<value_type>

        \param rhs
        std::initializer_list of value_type
        */
        /**************************************************************************/
        sllist(const std::initializer_list<value_type>& rhs);
        
        /***************************************************************************/
        /*!
        \brief
        Range Constructor

        \param start
        A const_pointer to the first element of the range

        \param end
        A const_pointer to one past the last element of the range
        */
        /**************************************************************************/
        sllist(const_pointer start, const_pointer end);

        /***************************************************************************/
        /*!
        \brief
        Destructor
        */
        /**************************************************************************/
        ~sllist();

        /***************************************************************************/
        /*!
        \brief
        Copy assignment with another sllist<T> object

        \param rhs
        Another sllist<T> object

        \return
        Reference to invoking sllist<T> object
        */
        /**************************************************************************/
        sllist& operator=(const sllist& rhs);

        /***************************************************************************/
        /*!
        \brief
        Copy assignment with std::initializer_list<value_type>

        \param rhs
        std::initializer_list of value_type

        \return
        Reference to invoking sllist<T> object
        */
        /**************************************************************************/
        sllist& operator=(const std::initializer_list<value_type>& rhs);

        /***************************************************************************/
        /*!
        \brief
        Concatenates with nodes of another sllist<T> container

        \param rhs
        Another sllist<T> object

        \return
        Reference to invoking sllist<T> object
        */
        /**************************************************************************/
        sllist& operator+=(const sllist& rhs);

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
        sllist& operator+=(const std::initializer_list<value_type>& rhs);

        /***************************************************************************/
        /*!
        \brief
        Returns data of node at position index in ListInt object

        \param index
        Position in sllist<T> object

        \return
        reference to data of node at position index in sllist<T> object
        */
        /**************************************************************************/
        reference operator[](size_type index);

        /***************************************************************************/
        /*!
        \brief
        Returns data of node at position index in sllist<T> object

        \param index
        Position in sllist<T> object

        \return
        const reference to data of node at position index in sllist<T> object
        */
        /**************************************************************************/
        const_reference operator[](size_type index) const;
        
        /***************************************************************************/
        /*!
        \brief
        Returns a reference to the first element in the sllist<T> container.

        \return
        A reference to the first element in the container.
        */
        /**************************************************************************/
        reference front();
        
        /***************************************************************************/
        /*!
        \brief
        Returns a const-reference to the first element in the sllist<T> container.

        \return
        A const-reference to the first element in the container.
        */
        /**************************************************************************/
        const_reference front() const;

        /***************************************************************************/
        /*!
        \brief
        Inserts a new element at the beginning of sllist<T>, right before its 
        current first element. The content of value is copied to the inserted element.

        \param value
        Value to be copied to the inserted element.
        */
        /**************************************************************************/
        void push_front(const_reference value);

        /***************************************************************************/
        /*!
        \brief
        Inserts a new element at the end of sllist<T>, right after its  current last 
        element. The content of value is copied to the inserted element.

        \param value
        Value to be copied to the inserted element.
        */
        /**************************************************************************/
        void push_back(const_reference value);

        /***************************************************************************/
        /*!
        \brief
        Removes the first element in the sllist<T> container, effectively reducing 
        its size by one.
        */
        /**************************************************************************/
        void pop_front();

        /***************************************************************************/
        /*!
        \brief
        Removes all elements from the sllist<T> container (which are destroyed), and 
        leaving the container with a size of 0
        */
        /**************************************************************************/
        void clear();

        /***************************************************************************/
        /*!
        \brief
        Exchanges the content of the container by the content of rhs, which is another
        sllist<T> object of the same type.

        \param rhs
        Another sllist<T> container of the same type
        */
        /**************************************************************************/
        void swap(sllist& rhs);

        /***************************************************************************/
        /*!
        \brief
        Returns number of nodes

        \return
        The number of nodes in a sllist<T> container
        */
        /**************************************************************************/
        size_type size() const;

        /***************************************************************************/
        /*!
        \brief
        Returns a bool value indicating whether the sllist<T> container is empty

        \return
        true if the container size is 0, false otherwise
        */
        /**************************************************************************/
        bool empty() const;  

        /***************************************************************************/
        /*!
        \brief
        Returns the number of sllist<T> objects active

        \return
        The number of sllist<T> objects active
        */
        /**************************************************************************/
        static size_type object_count();
        
        /***************************************************************************/
        /*!
        \brief
        Returns the number of Node objects active

        \return
        The number of Node objects active
        */
        /**************************************************************************/
        static size_type node_count();

    private:
        struct Node
        {
            Node *next{nullptr};
            value_type data;
            Node(value_type const&);
            ~Node();

            static size_type node_counter;
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
        Node* new_node(const_reference data) const;
    };

    /***************************************************************************/
    /*!
    \brief
    Add two sllist<T> objects

    \param lhs
    First sllist<T> object

    \param rhs
    Second sllist<T> object

    \return
    Result of adding two sllist<T> objects
    */
    /**************************************************************************/
    template <typename T>
    sllist<T> operator+(const sllist<T> &lhs, const sllist<T> &rhs);

    /***************************************************************************/
    /*!
    \brief
    Add sllist<T> object and std::initializer_list<ListInt::value_type>

    \param lhs
    sllist<T> object

    \param rhs
    std::initializer_list of T

    \return
    Result of adding sllist<T> object and std::initializer_list<T>
    */
    /**************************************************************************/
    template <typename T>
    sllist<T> operator+(const sllist<T> &lhs, const std::initializer_list<T> &rhs);

    /***************************************************************************/
    /*!
    \brief
    Add std::initializer_list<ListInt::value_type> and sllist<T> object

    \param lhs
    std::initializer_list of T

    \param rhs
    sllist<T> object

    \return
    Result of std::initializer_list<T> and sllist<T> object
    */
    /**************************************************************************/
    template <typename T>
    sllist<T> operator+(const std::initializer_list<T> &lhs, const sllist<T> &rhs);
    
    /***************************************************************************/
    /*!
    \brief
    Exchanges the content of lhs by the content of rhs, which are sllist<T>
    objects of the same type.

    \param lhs
    First sllist<T> container of the same type

    \param rhs
    Second sllist<T> container of the same type
    */
    /**************************************************************************/
    template <typename T>
    void swap(sllist<T> &lhs, sllist<T> &rhs);


template <typename T>
std::ostream& operator<<(std::ostream& os, sllist<T> const& list);

#include "sllist.tpp"

}

#endif

