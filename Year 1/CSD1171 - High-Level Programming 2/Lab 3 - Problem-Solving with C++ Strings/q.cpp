/*!************************************************************************
    \file q.cpp

    \author Bryan Ang Wei Ze

    \par DP email: bryanweize.ang\@digipen.edu

    \par Course: CSD1171 High-Level Programming 2

    \par Lab #3

    \date 01-26-2024

    \brief
    This source file implements a function inside namespace hlp2 that converts
    an English word into Pig Latin based on a simple set of rules.
**************************************************************************/
#include <string>

namespace
{
    /*!***********************************************************************
        \brief
        Definition of helper function that determines if a latter is a vowel

        \param[in] c
        Character to be analyzed

        \return
        True if the letter is a vowel, false otherwise
    **************************************************************************/
    bool is_vowel(char c)
    {
        if ('A' == c || 'a' == c || 'E' == c || 'e' == c || 'I'== c || 'i' == c ||
        'O' == c || 'o' == c || 'U' == c || 'u' == c)
        {
            return true;
        }

        return false;
    }

    /*!***********************************************************************
        \brief
        Definition of helper function that slices a word by moving the first
        letter to the back

        \param[in,out] word
        Word to be sliced
    **************************************************************************/
    void slice_word(std::string &word)
    {
        char first_letter{word.at(0)};
        word = word.substr(1) + first_letter;
    }
}

/*!************************************************************************
    \namespace hlp2
**************************************************************************/
namespace hlp2
{
    /*!***********************************************************************
        \brief
        Definition of function that converts an English word into Pig Latin 
        based on a simple set of rules.

        \param[in] word
        English word to be converted into Pig Latin

        \return
        The Pig Latin version of the English word
    **************************************************************************/
    std::string to_piglatin(std::string word)
    {
        bool contains_vowel{false};
        bool begins_with_capital_letter{false};
        
        if (is_vowel(word.at(0)))
        {
            word += "-yay";
        }
        else
        {
            if (isupper(word.at(0)))
            {
                begins_with_capital_letter = true;
                word.at(0) += 32;
            }
              
            for (size_t i{0}; i < word.size(); i++)
            {
                slice_word(word);
                char first_letter_new{word.at(0)};
                if (is_vowel(first_letter_new) || 'Y' == first_letter_new || 'y' == first_letter_new)
                {
                    contains_vowel = true;
                    word += "-ay";
                    break;
                }
            }

            if (!contains_vowel)
            {
                word += "-way";
            }

            if (begins_with_capital_letter)
            {
                word.at(0) -= 32;
            }
            
            
        }
        
        
        return word;
    }
}