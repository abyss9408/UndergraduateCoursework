/*!************************************************************************
    \file q.hpp

    \author Bryan Ang Wei Ze

    \par DP email: bryanweize.ang\@digipen.edu

    \par Course: CSD1171 High-Level Programming 2

    \par Lab #4

    \date 02-02-2024
    
    \brief
    This header file declares functions that implement a singly linked list
**************************************************************************/
#ifndef SLLIST_HPP
#define SLLIST_HPP
#include <cstddef>

/*!************************************************************************
    \namespace hlp2
**************************************************************************/
namespace hlp2
{
    struct node;
    struct sllist;
    
    /*!***********************************************************************
        \brief
        Declaration of function that accesses node's data

        \param[in] p
        Pointer to const node

        \return
        The value that the node contains
    **************************************************************************/
    int data(node const *p);

    /*!***********************************************************************
        \brief
        Declaration of function that modifies node's data

        \param[in, out] p
        Pointer to node

        \param[in] newval
        The new value
    **************************************************************************/
    void data(node *p, int newval);

    /*!***********************************************************************
        \brief
        Declaration of function that returns pointer to successor node

        \param[in, out] p
        Pointer to node

        \return
        Pointer to successor node
    **************************************************************************/
    node* next(node *p);

    /*!***********************************************************************
        \brief
        Declaration of function that returns pointer to const successor node

        \param[in] p
        Pointer to const node

        \return
        Pointer to const successor node
    **************************************************************************/
    node const* next(node const *p);

    /*!***********************************************************************
        \brief
        Declaration of function that create an unnamed object of type sllist
        on the free store

        \return
        Pointer to object of type sllist
    **************************************************************************/
    sllist* construct();

    /*!***********************************************************************
        \brief
        Declaration of function that deallocates memory allocated to list
        nodes and list object sllist

        \param[in, out] ptr_sll
        Pointer to sllist
    **************************************************************************/
    void destruct(sllist *ptr_sll);

    /*!***********************************************************************
        \brief
        Declaration of function that check if there are any nodes in a linked
        list

        \param[in] ptr_sll
        Pointer to const sllist

        \return
        True if there 0 nodes in the list pointed to by ptr_sll. Otherwise,
        false
    **************************************************************************/
    bool empty(sllist const *ptr_sll);

    /*!***********************************************************************
        \brief
        Declaration of function that counts the number of nodes in a linked
        list

        \param[in] ptr_sll
        Pointer to const sllist

        \return
        Number of nodes in the list pointed to by ptr_sll
    **************************************************************************/
    size_t size(sllist const *ptr_sll);

    /*!***********************************************************************
        \brief
        Declaration of function that adds a new node to the front of the linked
        list

        \param[in, out] ptr_sll
        Pointer to sllist

        \param[in] value
        Value of new node
    **************************************************************************/
    void push_front(sllist *ptr_sll, int value);

    /*!***********************************************************************
        \brief
        Declaration of function that adds a new node to the back of the linked
        list

        \param[in, out] ptr_sll
        Pointer to sllist

        \param[in] value
        Value of new node
    **************************************************************************/
    void push_back(sllist *ptr_sll, int value);

    /*!***********************************************************************
        \brief
        Declaration of function that deletes the first node encountered with
        the same value as second parameter value

        \param[in, out] ptr_sll
        Pointer to sllist

        \param[in] value
        Value of node to be deleted
    **************************************************************************/
    void remove_first(sllist *ptr_sll, int value);

    /*!***********************************************************************
        \brief
        Declaration of function that inserts a new node encapsulating data
        equal to parameter value into the list pointed to by parameter ptr_sll
        at an index specified by parameter index

        \param[in, out] ptr_sll
        Pointer to sllist

        \param[in] value
        Value of new node

        \param[in] index
        Position of new node
    **************************************************************************/
    void insert(sllist *ptr_sll, int value, size_t index);

    /*!***********************************************************************
        \brief
        Declaration of function that returns pointer to first node

        \param[in, out] ptr_sll
        Pointer to sllist

        \return
        Pointer to first node
    **************************************************************************/
    node* front(sllist *ptr_sll);

    /*!***********************************************************************
        \brief
        Declaration of function that returns pointer to const first node

        \param[in] ptr_sll
        Pointer to const sllist

        \return
        Pointer to const first node
    **************************************************************************/
    node const* front(sllist const *ptr_sll);

    /*!***********************************************************************
        \brief
        Declaration of function that returns a pointer to the first node in the
        list pointed to by parameter ptr_sll whose data is equal to the second
        parameter value 

        \param[in] ptr_sll
        Pointer to const sllist

        \param[in] value
        Value of node to find

        \return
        Pointer to first node with data equal to second parameter value 
    **************************************************************************/
    node* find(sllist const *ptr_sll, int value);
}
#endif