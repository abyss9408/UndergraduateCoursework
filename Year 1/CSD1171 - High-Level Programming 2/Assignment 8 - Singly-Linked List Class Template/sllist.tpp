/*!************************************************************************
    \file sllist.tpp

    \author Bryan Ang Wei Ze

    \par DP email: bryanweize.ang\@digipen.edu

    \par Course: CSD1171 High-Level Programming 2

    \par Programming Assignment #8

    \date 14-03-2024
    
    \brief
    This file contains definitions of ALL static data, static member,
    class template member and non-member functions of sllist<T>.
**************************************************************************/

  template <typename T> 
  typename sllist<T>::size_type sllist<T>::object_counter = 0;

  template <typename T> 
  typename sllist<T>::size_type sllist<T>::Node::node_counter = 0;

  template <typename T> 
  sllist<T>::Node::Node(const_reference rhs) : data{rhs}
  {
    
  }

  template <typename T> 
  sllist<T>::Node::~Node()
  {
    --Node::node_counter;
  }

  template <typename T> 
  typename sllist<T>::Node* sllist<T>::new_node(const_reference data) const
  {
    ++Node::node_counter;
    return new Node{data};
  }

  template <typename T> 
  sllist<T>::sllist() : head{nullptr}, tail{nullptr}, counter{0}
  {
    ++object_counter;
  }

  template <typename T> 
  typename sllist<T>::size_type sllist<T>::size() const
  {
    return counter;
  }

  template <typename T>
  bool sllist<T>::empty() const
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

  template <typename T> 
  typename sllist<T>::size_type sllist<T>::object_count()
  {
    return object_counter;
  }

  template <typename T> 
  typename sllist<T>::size_type sllist<T>::node_count()
  {
    return Node::node_counter;
  }

  template <typename T> 
  void sllist<T>::push_back(const_reference value)
  {

    Node *new_node = sllist<T>::new_node(value);
    
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

  template <typename T>
  sllist<T>::sllist(const sllist<T>& rhs) : counter{0}
  {
    Node *current_node{rhs.head};
    while (current_node)
    {
      push_back(current_node->data);
      current_node = current_node->next;
    }

    ++object_counter;
  }

  template <typename T>
  sllist<T>::sllist(const std::initializer_list<value_type>& rhs) : counter{0}
  {
    for (const_reference val : rhs)
    {
      push_back(val);
    }
    ++object_counter;
  }

  template <typename T>
  sllist<T>::sllist(const_pointer start, const_pointer end) : counter{0}
  {
    while (start != end)
    {
      push_back(*start);
      ++start;
    }

    ++object_counter;
  }

  template <typename T>
  void sllist<T>::clear()
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

        --counter;
      }
    }
  }

  template <typename T>
  sllist<T>::~sllist()
  {
    clear();
    --object_counter;
  }

  template <typename T>
  void sllist<T>::swap(sllist<T> &rhs)
  {
    std::swap(head, rhs.head);
    std::swap(tail, rhs.tail);
    std::swap(counter, rhs.counter);
  }

  template <typename T>
  sllist<T>& sllist<T>::operator=(const sllist<T>& rhs)
  {
    sllist<T> tmp{rhs};
    swap(tmp);
    return *this;
  }

  template <typename T>
  sllist<T>& sllist<T>::operator=(const std::initializer_list<value_type>& rhs)
  {
    sllist<T> tmp{rhs};
    swap(tmp);
    return *this;
  }

  template <typename T>
  typename sllist<T>::reference sllist<T>::operator[](size_type index)
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

  template <typename T>
  typename sllist<T>::const_reference sllist<T>::operator[](size_type index) const
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

  template <typename T>
  typename sllist<T>::reference sllist<T>::front()
  {
    return head->data;
  }

  template <typename T>
  typename sllist<T>::const_reference sllist<T>::front() const
  {
    return head->data;
  }

  template <typename T>
  sllist<T>& sllist<T>::operator+=(const sllist<T>& rhs)
  {
    sllist<T> lhs{*this};
    for (size_type i{}; i < rhs.counter; ++i)
    {
      push_back(rhs[i]);
    }

    return *this;
  }

  template <typename T>
  sllist<T>& sllist<T>::operator+=(const std::initializer_list<value_type>& rhs)
  {
    sllist<T> lhs{*this};
    for (const_reference val : rhs)
    {
      push_back(val);
    }
    
    return *this;
  }

  template <typename T>
  sllist<T> operator+(const sllist<T> &lhs, const sllist<T> &rhs)
  {
    sllist<T> result{lhs};
    result += rhs;
    return result;
  }

  template <typename T>
  sllist<T> operator+(const sllist<T> &lhs, const std::initializer_list<T> &rhs)
  {
    sllist<T> result{lhs};
    result += rhs;
    return result;
  }

  template <typename T>
  sllist<T> operator+(const std::initializer_list<T> &lhs, const sllist<T> &rhs)
  {
    sllist<T> result{lhs};
    result += rhs;
    return result;
  }

  template <typename T>
  void sllist<T>::push_front(const_reference value)
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
        
  template <typename T>
  void sllist<T>::pop_front()
  {
    Node* node_to_remove{head};
    head = node_to_remove->next;
    delete node_to_remove;
    --counter;
  }

template <typename T>
void swap(sllist<T> &lhs, sllist<T> &rhs)
{
  std::swap(lhs, rhs);
}

template <typename T>
std::ostream& operator<<(std::ostream& os, sllist<T> const& list)
{
  // start at the first node ...
  typename sllist<T>::size_type ls_sz = list.size();

  // uses overloaded subscript operator for class sllist ...
  for (typename sllist<T>::size_type i = 0; i < ls_sz; ++i) {
    os << list[i] << std::setw(4);
  }
  os << "\n"; // extra newline for readability
  return os;
}
