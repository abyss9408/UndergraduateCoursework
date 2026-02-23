/*!************************************************************************
  \file solution.hpp

  \author Bryan Ang Wei Ze

  \par DP email: bryanweize.ang\@digipen.edu

  \par Course: CSD2126 Modern C++ Design Patterns

  \par Programming Assignment #7

  \date 12-11-2024
  
  \brief
  This header file contains definitions of functions that process file records stats
**************************************************************************/
void print_file_names(const file_records& map)
{
    std::for_each(std::cbegin(map), std::cend(map), std::bind(print_file_name, 
    std::bind(split, std::placeholders::_1)));
}

size_t print_non_empty_files(const file_records& map)
{
    file_records temp;
    std::copy_if(std::cbegin(map), std::cend(map), std::inserter(temp, std::begin(temp)), 
    std::bind(check_if_empty, std::placeholders::_1, true));

    std::for_each(std::cbegin(temp), std::cend(temp), std::bind(print_file_name, 
    std::bind(split, std::placeholders::_1)));
    return std::size(temp);
}

size_t print_empty_files(const file_records& map)
{
    file_records temp;
    std::copy_if(std::cbegin(map), std::cend(map), std::inserter(temp, std::begin(temp)), 
    std::bind(check_if_empty, std::placeholders::_1, false));

    std::for_each(std::cbegin(temp), std::cend(temp), std::bind(print_file_name, 
    std::bind(split, std::placeholders::_1)));
    return std::size(temp);
}

std::tuple<file_records&> get_parameters(file_records& map)
{
    return std::forward_as_tuple(std::ref(map));
}

void remove_empty(file_records& map)
{
    file_records temp;
    std::copy_if(std::begin(map), std::end(map), std::inserter(temp, std::begin(temp)), 
    std::bind(check_if_empty, std::placeholders::_1, true));
    map = std::move(temp);
}