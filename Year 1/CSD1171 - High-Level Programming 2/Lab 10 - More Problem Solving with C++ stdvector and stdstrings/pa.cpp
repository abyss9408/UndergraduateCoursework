/**************************************************************************/
/*!
  \file pa.cpp

  \author Bryan Ang Wei Ze

  \par DP email: bryanweize.ang\@digipen.edu

  \par Course: CSD1171 High-Level Programming 2

  \par Lab 10

  \date 02-19-2024

  \brief
    This source file implements functions to create and print generalized
    checkerboards of characters
*/
/**************************************************************************/
#include "pa.hpp"

namespace hlp2
{
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
    Board create_board(std::vector<std::string> const& cmdline_params)
    {    
        Board new_board;
        
        // insufficient number of arguments
        if (cmdline_params.size() < 5)
        {
            std::cout << "Usage: ./program-name rows cols start cycle width\n";
            return Board{0,0,0,0};
        }
        
        // any of the integral parameters is negative
        if (std::stoi(cmdline_params[0]) <= 0 || std::stoi(cmdline_params[1]) <= 0 ||
        std::stoi(cmdline_params[3]) <= 0 || std::stoi(cmdline_params[4]) <= 0)
        {
            return Board{0,0,0,0};
        }
        
        // starting character ascii value plus cycles is greater than largest correct ascii value
        if (cmdline_params[2][0] + std::stoi(cmdline_params[3]) > 127)
        {
            return Board{0,0,0,0};
        }
        
        new_board.rows = std::stoi(cmdline_params[0]);
        new_board.columns = std::stoi(cmdline_params[1]);
        new_board.start = cmdline_params[2][0];
        new_board.cycles = std::stoi(cmdline_params[3]);

        return new_board;
    }
    
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
    void print_board(Board const& board, std::string const& w)
    {
        int width{};
        try
        {
            width = std::stoi(w);
        }
        catch (const std::invalid_argument&)
        {
            return;
        }
        for (int i = 0; i < board.rows; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                for (int k = 0; k < board.columns; ++k)
                {
                    for (int m = 0; m < width; ++m)
                    {
                        std::cout << static_cast<char>(board.start + (i + k) % board.cycles);
                    }
                    
                }
                std::cout << '\n';
            }
        }
        
    }
}