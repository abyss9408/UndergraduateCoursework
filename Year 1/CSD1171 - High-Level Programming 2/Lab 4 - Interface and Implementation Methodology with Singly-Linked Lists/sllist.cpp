/*!************************************************************************
    \file q.cpp

    \author Bryan Ang Wei Ze

    \par DP email: bryanweize.ang\@digipen.edu

    \par Course: CSD1171 High-Level Programming 2

    \par Lab #4

    \date 02-02-2024
    
    \brief
    This source file defines functions that implement a singly linked list
**************************************************************************/
#include "sllist.hpp"

namespace
{
    hlp2::node* create_node(int value, hlp2::node *next = nullptr);
}

/*!************************************************************************
    \namespace hlp2
**************************************************************************/
namespace hlp2
{
    struct node
    {
        int value;
        node *next;
    };
    
    /*!***********************************************************************
        \brief
        Definition of function that accesses node's data

        \param[in] p
        Pointer to const node

        \return
        The value that the node contains
    **************************************************************************/
    int data(node const *p){return p->value;}

    /*!***********************************************************************
        \brief
        Definition of function that modifies node's data

        \param[in, out] p
        Pointer to node

        \param[in] newval
        The new value
    **************************************************************************/
    void data(node *p, int newval){p->value = newval;}
    
    /*!***********************************************************************
        \brief
        Definition of function that returns pointer to successor node

        \param[in, out] p
        Pointer to node

        \return
        Pointer to successor node
    **************************************************************************/
    node* next(node *p){return p->next;}

    /*!***********************************************************************
        \brief
        Definition of function that returns pointer to const successor node

        \param[in] p
        Pointer to const node

        \return
        Pointer to const successor node
    **************************************************************************/
    node const* next(node const *p){return p->next;}

    struct sllist
    {
        node *head;
    };
    
    /*!***********************************************************************
        \brief
        Definition of function that create an unnamed object of type sllist
        on the free store

        \return
        Pointer to object of type sllist
    **************************************************************************/
    sllist* construct()
    {
        return new sllist{nullptr};
    }

    /*!***********************************************************************
        \brief
        Definition of function that deallocates memory allocated to list
        nodes and list object sllist

        \param[in, out] ptr_sll
        Pointer to sllist
    **************************************************************************/
    void destruct(sllist *ptr_sll)
    {
        // first node
        node *current_node{front(ptr_sll)};
        node *p;

        // if linked list is not empty
        if (!empty(ptr_sll))
        {
            while (current_node)
            {
                // p points to next node
                p = next(current_node);
                // make head node points to the next node
                ptr_sll->head = p;
                // delete the current node
                delete current_node;
                // navigate to the next node
                current_node = p;
            }
        }
        delete ptr_sll;
    }

    /*!***********************************************************************
        \brief
        Definition of function that check if there are any nodes in a linked
        list

        \param[in] ptr_sll
        Pointer to const sllist

        \return
        True if there 0 nodes in the list pointed to by ptr_sll. Otherwise,
        false
    **************************************************************************/
    bool empty(sllist const *ptr_sll)
    {
        if (ptr_sll->head == nullptr)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    /*!***********************************************************************
        \brief
        Definition of function that counts the number of nodes in a linked
        list

        \param[in] ptr_sll
        Pointer to const sllist

        \return
        Number of nodes in the list pointed to by ptr_sll
    **************************************************************************/
    size_t size(sllist const *ptr_sll)
    {
        size_t cnt {};
        for (node const *head = front(ptr_sll); head; head = next(head))
        {
            ++cnt;
        }
        return cnt;
    }

    /*!***********************************************************************
        \brief
        Declaration of function that adds a new node to the front of the linked
        list

        \param[in, out] ptr_sll
        Pointer to sllist

        \param[in] value
        Value of new node
    **************************************************************************/
    void push_front(sllist *ptr_sll, int value) 
    {
        ptr_sll->head = create_node(value, ptr_sll->head);
    }

    /*!***********************************************************************
        \brief
        Definition of function that adds a new node to the back of the linked
        list

        \param[in, out] ptr_sll
        Pointer to sllist

        \param[in] value
        Value of new node
    **************************************************************************/
    void push_back(sllist *ptr_sll, int value) 
    {   
        node *new_node = create_node(value, nullptr);
        if (empty(ptr_sll))
        {
            ptr_sll->head = new_node;
            return;
        }

        // get to the end of linked list
        node *last_node{ptr_sll->head};
        while (next(last_node))
        {
            last_node = next(last_node);
        }

        // get the last node to point to new node
        last_node->next = new_node;
    }

    /*!***********************************************************************
        \brief
        Definition of function that deletes the first node encountered with
        the same value as second parameter value

        \param[in, out] ptr_sll
        Pointer to sllist

        \param[in] value
        Value of node to be deleted
    **************************************************************************/
    void remove_first(sllist *ptr_sll, int value)
    {
        node *p{ptr_sll->head};
        node *q;

        // if linked list in not empty
        if (!empty(ptr_sll))
        {
            if (p->value == value)
            {
                ptr_sll->head = next(p);
                delete p;
                return;
            }
        
            while (next(p))
            {
                if (p->next->value == value)
                {
                    q = next(p);
                    p->next = next(q);
                    delete q;
                    return;
                }
                p = next(p);
            }
        } 
    }

    /*!***********************************************************************
        \brief
        Definition of function that inserts a new node encapsulating data
        equal to parameter value into the list pointed to by parameter ptr_sll
        at an index specified by parameter index

        \param[in, out] ptr_sll
        Pointer to sllist

        \param[in] value
        Value of new node

        \param[in] index
        Position of new node
    **************************************************************************/
    void insert(sllist *ptr_sll, int value, size_t index)
    {
        node *new_node = create_node(value, nullptr);
        node *current_node{ptr_sll->head};
        size_t current_index{};
        bool exist{false};

        // insert the new node in front of list if index is 0
        if (index == 0)
        {
            new_node->next = front(ptr_sll);
            ptr_sll->head = new_node;
            return;
        }
        
        // index != 0
        while (next(current_node))
        {
            if (current_index == index - 1)
            {
                new_node->next = next(current_node);
                current_node->next = new_node;
                exist = true;
                break;
            }
            current_node = current_node->next;
            current_index++;
        }
        
        // the position does not exist
        if (!exist)
        {
            current_node->next = new_node;
        }
    }

    /*!***********************************************************************
        \brief
        Definition of function that returns pointer to first node

        \param[in, out] ptr_sll
        Pointer to sllist

        \return
        Pointer to first node
    **************************************************************************/
    node* front(sllist *ptr_sll)
    {
        return ptr_sll->head;
    }

    /*!***********************************************************************
        \brief
        Declaration of function that returns pointer to const first node

        \param[in] ptr_sll
        Pointer to const sllist

        \return
        Pointer to const first node
    **************************************************************************/
    node const* front(sllist const *ptr_sll)
    {
        return ptr_sll->head;
    }

    /*!***********************************************************************
        \brief
        Definition of function that returns a pointer to the first node in the
        list pointed to by parameter ptr_sll whose data is equal to the second
        parameter value 

        \param[in] ptr_sll
        Pointer to const sllist

        \param[in] value
        Value of node to find

        \return
        Pointer to first node with data equal to second parameter value 
    **************************************************************************/
    node* find(sllist const *ptr_sll, int value)
    {
        // first node
        node *find {ptr_sll->head};

        // navigate through the linked list
        while (find)
        {
            if (data(find) == value)
            {
                return find;
            }
            // next node
            find = next(find);
        }
        
        return nullptr;
    }

    
}

namespace
{
    hlp2::node* create_node(int value, hlp2::node *next)
    {
        return new hlp2::node {value, next};
    }
}
