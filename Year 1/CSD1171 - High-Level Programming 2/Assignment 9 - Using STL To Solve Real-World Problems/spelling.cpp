// provide file header documentation block

#include "spelling.hpp"

// include standard header files required to compile this file [that are not included in spelling.hpp!!!
#include <fstream>

namespace hlp2 {

// define member functions of class string_utils
// define member functions of class spell_checker
spell_checker::spell_checker(const std::string &lexicon) 
: dictionary{lexicon}
{

}

spell_checker::SCResult spell_checker::words_starting_with(char letter, size_t& count) const
{
    std::ifstream ifs;
    std::string word;

    ifs.open(dictionary);
    if (!ifs)
    {
        return scrFILE_ERR_OPEN;
    }

    letter = std::toupper(letter);

    while (ifs >> word)
    {
        word[0] = std::toupper(word[0]);

        if (word[0] == letter)
        {
            ++count;
        }
    }

    return scrFILE_OK;
}

spell_checker::SCResult spell_checker::word_lengths(std::vector<size_t>& lengths, size_t count) const
{
    std::ifstream ifs;
    std::string word;

    ifs.open(dictionary);
    if (!ifs)
    {
        return scrFILE_ERR_OPEN;
    }

    while (ifs >> word)
    {
        if (word.size() <= count)
        {
            ++lengths[word.size()];
        }
    }

    return scrFILE_OK;
}

spell_checker::SCResult spell_checker::get_info(lexicon_info& info) const
{
    std::ifstream ifs;
    std::string word;
    size_t word_count{}, shortest{}, longest{};

    ifs.open(dictionary);
    if (!ifs)
    {
        return scrFILE_ERR_OPEN;
    }

    // read in first word and assume that it's the longest and shortest word
    ifs >> word;
    shortest = word.size();
    longest = word.size();

    ++word_count;

    // read in the rest of the words
    while (ifs >> word)
    {
        if (word.size() > longest)
        {
            longest = word.size();
        }

        if (word.size() < shortest)
        {
            shortest = word.size();
        }

        ++word_count;
    }

    info.count = word_count;
    info.longest = longest;
    info.shortest = shortest;

    return scrFILE_OK;
}

spell_checker::SCResult spell_checker::spellcheck(std::string const& word) const
{
    std::ifstream ifs;
    std::string read_word, compare{word};

    ifs.open(dictionary);
    if (!ifs)
    {
        return scrFILE_ERR_OPEN;
    }

    compare = string_utils::upper_case(compare);

    while (ifs >> read_word)
    {
        read_word = string_utils::upper_case(read_word);

        if (read_word == compare)
        {
            return scrWORD_OK;
        }

        /*if not found and current word is lexicographically greater than the word to search for,
        stop reading from lexicon*/ 
        if (read_word > compare)
        {
            break;
        }
    }

    return scrWORD_BAD;
}

spell_checker::SCResult spell_checker::acronym_to_word(std::string const& acronym, 
    std::vector<std::string>& words, size_t maxlen) const
{
    std::ifstream ifs;
    std::string read_word, read_word_all_uppercase;
    

    ifs.open(dictionary);
    if (!ifs)
    {
        return scrFILE_ERR_OPEN;
    }

    while (ifs >> read_word)
    {
        bool all_consec_letters_found{true};
        size_t pos{};

        read_word_all_uppercase = string_utils::upper_case(read_word);

        if (read_word_all_uppercase[0] == toupper(acronym[0]))
        {
            for (size_t i{1}; i < acronym.size(); ++i)
            {
                pos = read_word_all_uppercase.find_first_of(toupper(acronym[i]), pos+1);
                if (pos == std::string::npos)
                {
                    all_consec_letters_found = false;
                    break;
                }
            }

            if (all_consec_letters_found && (maxlen == 0 || read_word.size() <= maxlen))
            {
                words.push_back(read_word);
            }
        }
    }

    return scrFILE_OK;
}

std::string string_utils::upper_case(std::string const& str)
{
    std::string copy{str};

    for (char& ch : copy)
    {
        ch = toupper(ch);
    }
    
    return copy;
}

std::vector<std::string> string_utils::split(std::string const& words)
{
    size_t i{}, j{};
    std::string word;
    std::vector<std::string> words_in_line;

    while (true)
    {
        i = words.find_first_not_of(' ', i);
        j = words.find_first_of(' ', i);
        
        if (i == std::string::npos)
        {
            break;
        }

        word = words.substr(i, j-i);
        words_in_line.push_back(word);
        i = j;
    }
    
    return words_in_line;
}

} // end namespace hlp2
