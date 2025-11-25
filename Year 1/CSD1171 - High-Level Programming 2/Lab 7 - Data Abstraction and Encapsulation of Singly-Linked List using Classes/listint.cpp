/*!************************************************************************
    \file listint.cpp

    \author Bryan Ang Wei Ze

    \par DP email: bryanweize.ang\@digipen.edu

    \par Course: CSD1171 High-Level Programming 2

    \par Lab #7

    \date 08-03-2024
    
    \brief
    This source file implements constructors, member function operator overloads
    and non-member function operator overloads for the class ListInt
**************************************************************************/

#include "listint.hpp"


// This function definition is provided:
////////////////////////////////////////////////////////////////////////////////
// Function: operator<<
//  Purpose: Output the list into an ostream object
//   Inputs: os - ostream object
//           list - the List to output
//  Outputs: Returns an ostream object.
////////////////////////////////////////////////////////////////////////////////
namespace hlp2
{
  ListInt::size_type ListInt::object_counter = 0;

  ListInt::Node* ListInt::new_node(value_type data) const
  {
    return new Node{data, nullptr};
  }

  ListInt::ListInt() : head{nullptr}, tail{nullptr}, counter{0}
  {
    ++object_counter;
  }

  ListInt::size_type ListInt::size() const
  {
    return counter;
  }

  bool ListInt::empty() const
  {
    if (head == nullptr)
    {
      return true;
    }
    else
    {
      return false;
    }
  }

  ListInt::size_type ListInt::object_count()
  {
    return object_counter;
  }

  void ListInt::push_back(value_type value)
  {

    Node *new_node = ListInt::new_node(value);
    
    if (empty())
    {
      head = new_node;
      tail = new_node;
      ++counter;
      return;
    }
    
    tail->next = new_node;
    tail = new_node;
    ++counter;
  }

  ListInt::ListInt(const ListInt& rhs) : counter{0}
  {
    Node *current_node{rhs.head};
    while (current_node)
    {
      push_back(current_node->data);
      current_node = current_node->next;
    }

    ++object_counter;
  }

  ListInt::ListInt(const std::initializer_list<value_type>& rhs) : counter{0}
  {
    for (const value_type &val : rhs)
    {
      push_back(val);
    }
    ++object_counter;
  }

  void ListInt::clear()
  {
    // first node
    Node *current_node{head};

    Node *p;

    // if linked list is not empty
    if (!empty())
    {
      while (current_node)
      {
        // p points to next node
        p = current_node->next;
        // make head node points to the next node
        head = p;
        // delete the current node
        delete current_node;
        // navigate to the next node
        current_node = p;
      }
    }
  }

  ListInt::~ListInt()
  {
    clear();
    --object_counter;
  }

  void ListInt::swap(ListInt &rhs)
  {
    std::swap(head, rhs.head);
    std::swap(tail, rhs.tail);
    std::swap(counter, rhs.counter);
  }

  ListInt& ListInt::operator=(const ListInt& rhs)
  {
    ListInt tmp{rhs};
    swap(tmp);
    return *this;
  }

  ListInt& ListInt::operator=(const std::initializer_list<value_type>& rhs)
  {
    ListInt tmp{rhs};
    swap(tmp);
    return *this;
  }

  ListInt::reference ListInt::operator[](size_type index)
  {
    size_type current_index{};
    Node *current_node{head};

    while (current_index != index)
    {
      current_node = current_node->next;
      ++current_index;
    }
    
    return current_node->data;
  }

  ListInt::const_reference ListInt::operator[](size_type index) const
  {
    size_type current_index{};
    Node *current_node{head};

    while (current_index != index)
    {
      current_node = current_node->next;
      ++current_index;
    }
    
    return current_node->data;
  }

  ListInt& ListInt::operator+=(const ListInt& rhs)
  {
    for (size_type i{}; i < rhs.counter; ++i)
    {
      push_back(rhs[i]);
    }

    return *this;
  }

  ListInt& ListInt::operator+=(const std::initializer_list<value_type>& rhs)
  {
    for (const_reference val : rhs)
    {
      push_back(val);
    }
    
    return *this;
  }

  ListInt operator+(const ListInt &lhs, const ListInt &rhs)
  {
    ListInt result{lhs};
    result += rhs;
    return result;
  }

  ListInt operator+(const ListInt &lhs, const std::initializer_list<ListInt::value_type> &rhs)
  {
    ListInt result{lhs};
    result += rhs;
    return result;
  }

  ListInt operator+(const std::initializer_list<ListInt::value_type> &lhs, const ListInt &rhs)
  {
    ListInt result{lhs};
    result += rhs;
    return result;
  }

  void ListInt::push_front(value_type value)
  {
    Node* front_node = new_node(value);

    if (empty())
    {
      head = front_node;
      tail = front_node;
      ++counter;
      return;
    }
    
    front_node->next = head;
    head = front_node;
    ++counter;
  }
        
  ListInt::value_type ListInt::pop_front()
  {
    Node* node_to_remove{head};
    value_type val{node_to_remove->data};
    head = node_to_remove->next;
    delete node_to_remove;
    --counter;
    return val;
  }

  std::ostream& operator<<(std::ostream& os, const ListInt& rhs) {
  // Start at the first node
  ListInt::Node *pnode = rhs.head;

  // Until we reach the end of the list
  while (pnode != 0) {
    os << std::setw(4) << pnode->data; // print the data in this node
    pnode = pnode->next;               // move to the next node
  }
  
  os << std::endl; // extra newline for readability
  return os;
}
}


