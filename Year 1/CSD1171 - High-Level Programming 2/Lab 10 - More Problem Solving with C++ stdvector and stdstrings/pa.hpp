/**************************************************************************/
/*!
  \file pa.hpp

  \author Bryan Ang Wei Ze

  \par DP email: bryanweize.ang\@digipen.edu

  \par Course: CSD1171 High-Level Programming 2

  \par Lab 10

  \date 02-19-2024

  \brief
    This header file declares functions to create and print generalized
    checkerboards of characters
*/
/**************************************************************************/
#include <vector>
#include <string>
#include <iostream>

namespace hlp2
{
    struct Board
    {
        int rows;
        int columns;
        char start;
        int cycles;
    };
    
    /**************************************************************************/
        /*!
        \brief
        Creates and initializes an object of type Board
      
        \param cmdline_params
        Rows, columns, starting character and cycle size

        \return
        The Board object
        */
    /**************************************************************************/
    Board create_board(std::vector<std::string> const& cmdline_params);

    /**************************************************************************/
        /*!
        \brief
        Prints the characters of the board according to the supplied cmd line
        arguments
      
        \param board
        Board object that contains the parameters: rows, columns, starting
        character and cycle size

        \param width
        Each board element is printed as a width*width square
        */
    /**************************************************************************/
    void print_board(Board const& board, std::string const& width);
}