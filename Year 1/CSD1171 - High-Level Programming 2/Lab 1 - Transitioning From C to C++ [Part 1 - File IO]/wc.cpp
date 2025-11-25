/*!************************************************************************
\file wc.cpp
\author Bryan Ang Wei Ze
\par DP email: bryanweize.ang\@digipen.edu
\par Course: CSD1171 High-Level Programming 2
\par Lab #1
\date 01-12-2024
\brief
This source file implements a function inside namespace hlp2 that emulates
Linux utility wc that prints the number of lines terminated with terminating
character, number of words and characters
**************************************************************************/
#include <iostream>
#include <iomanip>
#include <fstream>

namespace {
  const size_t MAX_LINE_LEN {2048};
}

/*!************************************************************************
\namespace hlp2
**************************************************************************/
namespace hlp2 {
/*!***********************************************************************
\brief
Implementation of function that emulates Linux utility wc that prints the
number of lines terminated with terminating character, number of words and
characters.
\param[in] argc
The number of arguements supplied to command line
\param[in] argv
An array of file names supplied to command line including name of program
and other files
**************************************************************************/
  void wc(int argc, char *argv[])
  {
    int total_line_count{0}, total_word_count{0}, total_byte_count{0};
    for (int i = 1; i < argc; i++)
    {
      int line_count{0}, word_count{0}, byte_count{0};
      std::ifstream ifs{argv[i]};
      char line[MAX_LINE_LEN];

      // count number of words
      while (ifs >> line)
      {
        word_count++;
      }

      // clear eof
      ifs.clear();
      // go back to the beginning of file stream
      ifs.seekg(std::ios_base::beg);
      char read_char;

      // count number of lines with '\n' terminator and characters
      while (ifs.get(read_char))
      {
        if (read_char == '\n')
        {
          ++line_count;
        }
        ++byte_count;
      }
      ifs.close();
      std::cout <<std::setw(7) << line_count<<std::setw(8)<<word_count<<
      std::setw(8)<<byte_count<<" "<<argv[i]<<'\n';
      total_line_count+=line_count;
      total_word_count+=word_count;
      total_byte_count+=byte_count;
    }
    // print total if more than one file is being input
    if (argc > 2)
    std::cout<<std::setw(7)<<total_line_count<<std::setw(8)<<total_word_count<<
    std::setw(8)<<total_byte_count<<" total\n";
  }
} // end namespace hlp2
