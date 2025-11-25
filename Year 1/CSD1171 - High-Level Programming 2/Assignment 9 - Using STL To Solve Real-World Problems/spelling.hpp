/*!************************************************************************
    \file spelling.hpp

    \author Bryan Ang Wei Ze

    \par DP email: bryanweize.ang\@digipen.edu

    \par Course: CSD1171 High-Level Programming 2

    \par Programming Assignment #9

    \date 20-03-2024
    
    \brief
    This header file defines a spell-checker class and declares member functions
    that will be used to check words for correct spelling
**************************************************************************/

#ifndef SPELLING_HPP
#define SPELLING_HPP

// include only those standard header files that are referenced in this file!!!
#include <cstddef>
#include <string>
#include <vector>

namespace hlp2 {

// define class string_utils and class spell_checker AND any other structures/classes you wish ...
class spell_checker
{
public:
    enum SCResult 
    {
        scrFILE_OK = -1,
        scrFILE_ERR_OPEN = -2,
        scrWORD_OK = 1,
        scrWORD_BAD = 2
    };

    struct lexicon_info
    {
        size_t shortest;
        size_t longest;
        size_t count;
    };
    
    /***************************************************************************/
        /*!
        \brief
        Single-argument Constructor that takes in a lexicon file name

        \param lexicon
        lexicon file name
        */
    /**************************************************************************/
    spell_checker(const std::string &lexicon);
    
    /***************************************************************************/
        /*!
        \brief
        Count number of words that start with letter

        \param letter
        The character to search for

        \param
        The number of words the start with letter

        \return
        Result of Spell Check
        */
    /**************************************************************************/
    SCResult words_starting_with(char letter, size_t& count) const;
    
    /***************************************************************************/
        /*!
        \brief
        Count the number of words that have length 1 to count and store
        them in lengths at appropriate index

        \param lengths
        A container the store the number of words with respective lengths

        \param count
        Max length

        \return
        Result of Spell Check
        */
    /**************************************************************************/
    SCResult word_lengths(std::vector<size_t>& lengths, size_t count) const;
    
    /***************************************************************************/
        /*!
        \brief
        Return some information about lexicon using reference parameter

        \param info
        Lengths of shortest word, length of longest word and number of words in
        lexicon

        \return
        Result of Spell Check
        */
    /**************************************************************************/
    SCResult get_info(lexicon_info& info) const;
    
    /***************************************************************************/
        /*!
        \brief
        Lookup the word in lexicon

        \param word
        Word to be searched for

        \return
        Result of Spell Check
        */
    /**************************************************************************/
    SCResult spellcheck(std::string const& word) const;
    
    /***************************************************************************/
        /*!
        \brief
        Find words in the lexicon that are composed of letters in the same order

        \param acronym
        Acronym to be compared with

        \param words
        A container that contains words that are of the acronym

        \param maxlen
        Maxinum of length of words to search for

        \return
        Result of Spell Check
        */
    /**************************************************************************/
    SCResult acronym_to_word(std::string const& acronym, 
    std::vector<std::string>& words, size_t maxlen = 0) const;

private:
    std::string dictionary;
};

class string_utils
{
public:
    static std::string upper_case(std::string const& str);

    static std::vector<std::string> split(std::string const& words);
};

} // end namespace hlp2

#endif