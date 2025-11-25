/*!************************************************************************
\file q.cpp
\author Bryan Ang Wei Ze
\par DP email: bryanweize.ang\@digipen.edu
\par Course: CSD1171 High-Level Programming 2
\par Programming Assignment #2
\date 01-20-2024
\brief
This source file implements a function inside namespace hlp2 that adds
steganography to a message from an input file
**************************************************************************/
#include <string>
#include <fstream>
#include <iostream>

/*!************************************************************************
\namespace hlp2
**************************************************************************/
namespace hlp2 {
/*!***********************************************************************
\brief
Implementation of function that adds steganography to a message from an
input file
\param[in] filename
The file with the message that steganography will be added to
\param[in] keywords
Keywords for the steganography algorithm 
**************************************************************************/
  void extract(char const *filename, char const **keywords)
  {
    std::ifstream read_in(filename);
    std::string prev_word;
    std::string current_word;
    std::string next_word;
    bool is_prev_word_key{false};
    bool is_curr_word_key{false};

    // Check if input file exists
    if (!read_in.is_open())
    {
      std::cout << "File "<< filename <<" not found."<<'\n';
      return;
    }

    // read first three words
    read_in >> prev_word;
    read_in >> current_word;
    read_in >> next_word;
  
    while (read_in.good())
    {
      for (int i = 0; keywords[i] != nullptr; i++)
      {
        if (prev_word == keywords[i])
        {
          is_prev_word_key = true;
          break;
        }
        else
        {
          is_prev_word_key = false;
        }
      }

      for (int i = 0; keywords[i] != nullptr; i++)
        {
          if (current_word == keywords[i])
          {
            is_curr_word_key = true;
            break;
          }
          else
          {
            is_curr_word_key = false;
          }
        }
      
      /* Only output the next word if previous word is not a keyword
      and current word is a keyword*/
      if(!is_prev_word_key && is_curr_word_key)
      {
        std::cout << next_word<<' ';
      }
      
      prev_word = current_word;
      current_word = next_word;

      // Read in a new word
      read_in >> next_word;
    }
    read_in.close();
    std::cout << '\n';
  }
}
