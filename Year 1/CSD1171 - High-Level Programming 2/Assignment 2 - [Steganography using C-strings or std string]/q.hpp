/*!************************************************************************
\file q.hpp
\author Bryan Ang Wei Ze
\par DP email: bryanweize.ang\@digipen.edu
\par Course: CSD1171 High-Level Programming 2
\par Programming Assignment #2
\date 01-20-2024
\brief
This header file declares a function inside namespace hlp2 that adds
steganography to a message from an input file
**************************************************************************/
#ifndef Q_HPP_
#define Q_HPP_

/*!************************************************************************
\namespace hlp2
**************************************************************************/
namespace hlp2 {
/*!***********************************************************************
\brief
Declaration of function that adds steganography to a message from an
input file
\param[in] filename
The file with the message that steganography will be added to
\param[in] keywords
Keywords for the steganography algorithm 
**************************************************************************/
  void extract(char const *filename, char const **keywords);
}

#endif
