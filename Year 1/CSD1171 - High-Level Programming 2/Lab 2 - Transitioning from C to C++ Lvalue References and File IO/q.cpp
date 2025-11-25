/*!************************************************************************
\file q.cpp
\author Bryan Ang Wei Ze
\par DP email: bryanweize.ang\@digipen.edu
\par Course: CSD1171 High-Level Programming 2
\par Lab #2
\date 01-19-2024
\brief
This source file defines two functions inside namespace hlp2 that reads
tsunami data from text files and prints the data respectively and a helper
function that trims strings of leading and trailing spaces.
**************************************************************************/
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include "q.hpp"

namespace {
/*!***********************************************************************
\brief
Definition of function that trims strings of leading and trailing spaces
\param[in] s
The string to be trimmed
\return std::string
The trimmed string
**************************************************************************/
  std::string trim(const std::string &s)
  {
    size_t first = s.find_first_not_of(' ');
    if (std::string::npos == first)
    {
      return s;
    }
    size_t last = s.find_last_not_of(' ');
    return s.substr(first, (last - first + 1));
  }
}

/*!************************************************************************
\namespace hlp2
**************************************************************************/
namespace hlp2 {
/*!***********************************************************************
\brief
Definition of function that reads tsunami data from text files
\param[in] file_name
The name of the file to be read
\param[in, out] max_cnt
The number of tsunami events
\return Tsunami*
A pointer to the dynamically allocated array of tsunami data
**************************************************************************/
  Tsunami* read_tsunami_data(std::string const& file_name, int& max_cnt)
  {
    std::ifstream ifs(file_name);
    char read_char{'\0'};
    max_cnt = 0;
    Tsunami* list;

    if (!ifs.is_open())
    {
      return nullptr;
    }

    while (ifs.get(read_char))
    {
      if (read_char == '\n')
      {
        max_cnt++;
      }
    }

    ifs.clear();
    ifs.seekg(std::ios_base::beg);

    list = new Tsunami[max_cnt];

    for (int i = 0; i < max_cnt; i++)
    {
      ifs >> list[i].month;
      ifs >> list[i].day;
      ifs >> list[i].year;
      ifs >> list[i].fatalities;
      ifs >> list[i].max_wave_height;
      std::getline(ifs, list[i].location);
      list[i].location = trim(list[i].location);
    }
    
    ifs.close();
    
    return list;
  }

/*!***********************************************************************
\brief
Definition of function that prints tsunami data read by read_tsunami_data
\param[in] arr
The array of tsunami data
\param[in] size
The number of tsunami events
\param[in] file_name
The name of the file to output the tsunami data to
**************************************************************************/
  void print_tsunami_data(Tsunami const *arr, int size, 
  std::string const& file_name)
  {
    std::ofstream ofs(file_name);
    double max_height{arr[0].max_wave_height};
    double total_height{0};
    double average_height{0};
    ofs << "List of tsunamis:\n";
    ofs << "-----------------\n";
    for (int i = 0; i < size; i++)
    {
      ofs << std::setw(2) << std::setfill('0') << arr[i].month << ' ';
      ofs << std::setw(2) << arr[i].day << ' ';
      ofs << std::setw(4) << arr[i].year;
      ofs << std::setw(7) << std::setfill(' ') << arr[i].fatalities;
      ofs << std::fixed << std::setw(11) << std::setprecision(2) << 
      arr[i].max_wave_height << "     ";
      ofs << arr[i].location<<'\n';
    }
    
    ofs << '\n';
    ofs << "Summary information for tsunamis\n";
    ofs << "--------------------------------\n\n";

    for (int i = 1; i < size; i++)
    {
      if (arr[i].max_wave_height > max_height)
      {
        max_height = arr[i].max_wave_height;
      }
      
    }
    
    ofs << "Maximum wave height (in meters):" << std::fixed << std::setw(6) << 
    std::setprecision(2) <<max_height<<'\n';
    ofs << '\n';
    for (int i = 0; i < size; i++)
    {
       total_height += arr[i].max_wave_height;
    }
    average_height = total_height / size;
    ofs << "Average wave height (in meters):" << std::fixed << std::setw(6) << 
    std::setprecision(2) << average_height<<'\n';
    ofs << '\n';
    ofs << "Tsunamis with greater than average height "<< average_height<<":\n";
    for (int i = 0; i < size; i++)
    {
       if (arr[i].max_wave_height > average_height)
       {
        ofs << arr[i].max_wave_height << "     ";
        ofs << arr[i].location << '\n';
       }
       
    }
    ofs.close();
  }
}