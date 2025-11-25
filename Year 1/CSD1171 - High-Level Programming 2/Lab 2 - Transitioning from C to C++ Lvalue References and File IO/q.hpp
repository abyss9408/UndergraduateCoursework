/*!************************************************************************
\file q.hpp
\author Bryan Ang Wei Ze
\par DP email: bryanweize.ang\@digipen.edu
\par Course: CSD1171 High-Level Programming 2
\par Lab #2
\date 01-19-2024
\brief
This header file declares two functions inside namespace hlp2 that reads
tsunami data from text files and prints the data respectively.
**************************************************************************/
#ifndef Q_HPP
#define Q_HPP

#include <string>

/*!************************************************************************
\namespace hlp2
**************************************************************************/
namespace hlp2 {
/*!************************************************************************
\struct Tsunami
**************************************************************************/
  struct Tsunami {
    int month;
    int day;
    int year;
    int fatalities;
    double max_wave_height;
    std::string location;
  };
  
/*!***********************************************************************
\brief
Declaration of function that reads tsunami data from text files
\param[in] file_name
The name of the file to be read
\param[in, out] max_cnt
The number of tsunami events
\return Tsunami*
A pointer to the dynamically allocated array of tsunami data
**************************************************************************/
  Tsunami* read_tsunami_data(std::string const& file_name, int& max_cnt);

/*!***********************************************************************
\brief
Declaration of function that prints tsunami data read by read_tsunami_data
\param[in] arr
The array of tsunami data
\param[in] size
The number of tsunami events
\param[in] file_name
The name of the file to output the tsunami data to
**************************************************************************/
  void print_tsunami_data(Tsunami const *arr, int size,
  std::string const& file_name);
  
} // end namespace hlp2
#endif
