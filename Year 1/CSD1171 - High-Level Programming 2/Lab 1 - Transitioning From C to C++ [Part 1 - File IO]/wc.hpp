/*!************************************************************************
\file wc.hpp
\author Bryan Ang Wei Ze
\par DP email: bryanweize.ang\@digipen.edu
\par Course: CSD1171 High-Level Programming 2
\par Lab #1
\date 01-12-2024
\brief
This header file declares a function inside namespace hlp2 that emulates
Linux utility wc that prints the number of lines terminated with terminating
character, number of words and characters
**************************************************************************/
#ifndef WC_HPP
#define WC_HPP


/*!************************************************************************
\namespace hlp2
**************************************************************************/
namespace hlp2 {

/*!***********************************************************************
\brief
Declaration of function that emulates Linux utility wc that prints the
number of lines terminated with terminating character, number of words and
characters.
\param[in] argc
The number of arguements supplied to command line
\param[in] argv
An array of file names supplied to command line including name of program
and other files
**************************************************************************/
void wc(int argc, char *argv[]);
} // end namespace hlp2

#endif // end header guard
