/*!************************************************************************
\file q.hpp
\author Bryan Ang Wei Ze
\par DP email: bryanweize.ang\@digipen.edu
\par Course: CSD1171 High-Level Programming 2
\par Programming Assignment #1
\date 01-12-2024
\brief
This header file declares a function inside namespace hlp2 that analyze a
file and output an analysis file about the statistics of the input file
**************************************************************************/
#ifndef Q_HPP_
#define Q_HPP_


/*!************************************************************************
\namespace hlp2
**************************************************************************/
namespace hlp2 {
  /*!***********************************************************************
\brief
Declaration of function that computes total number of characters, number of
letters, white spaces, digits, characters, respective letters, integers,
sum of integers and average of integers
\param[in] input_filename
The file to be analyzed.
\param[in] analysis_file
The name of the analysis file that will be created based on the computed 
statistics of the input file.
**************************************************************************/
  void q(char const *input_filename, char const *analysis_file);
}

#endif
