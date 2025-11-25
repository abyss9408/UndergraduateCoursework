/*!************************************************************************
    \file q.hpp

    \author Bryan Ang Wei Ze

    \par DP email: bryanweize.ang\@digipen.edu

    \par Course: CSD1171 High-Level Programming 2

    \par Lab #3

    \date 01-26-2024
    
    \brief
    This header file declares a function inside namespace hlp2 that converts
    an English word into Pig Latin based on a simple set of rules.
**************************************************************************/
#ifndef Q_HPP
#define Q_HPP
#include <string>

/*!************************************************************************
    \namespace hlp2
**************************************************************************/
namespace hlp2
{
    /*!***********************************************************************
        \brief
        Declaration of function that converts an English word into Pig Latin 
        based on a simple set of rules.

        \param[in] word
        The English word to be converted into Pig Latin

        \return Pig Latin word
        The Pig Latin version of the English word
    **************************************************************************/
    std::string to_piglatin(std::string word);
}
#endif