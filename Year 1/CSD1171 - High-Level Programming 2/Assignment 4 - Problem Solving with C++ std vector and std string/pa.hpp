/**************************************************************************/
/*!
  \file pa.hpp

  \author Bryan Ang Wei Ze

  \par DP email: bryanweize.ang\@digipen.edu

  \par Course: CSD1171 High-Level Programming 2

  \par Programming Assignment #4

  \date 02-02-2024

  \brief
    This header file declares functions that read, process and transform an input
    file that has a table of names and populations of countries into four output
    files containing tables with countries names sorted in increasing and decreasing
    orders and population counts sorted in increasing and decreasing orders. 
    
*/
/**************************************************************************/
#ifndef PA_HPP
#define PA_CPP

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>

namespace hlp2
{
    struct CountryInfo
    {
        std::string name;
        long int pop;
    };
    
    using Ptr_Cmp_Func = bool (*)(CountryInfo const&, CountryInfo const&);
    
    /**************************************************************************/
    /*!
      \brief
        Read unsorted data of all countries from input stream into CountryInfo
        then store it into a vector
      
      \param is
        Input stream

      \return
        A vector that contains unsorted data of all countries from input stream
      
    */
    /**************************************************************************/
    std::vector<CountryInfo> fill_vector_from_istream(std::istream& is);

    /**************************************************************************/
    /*!
      \brief
         Determines the length of the longest country's name
      
      \param v
        A vector that contains unsorted data of all countries 

      \return
        The length of the longest country's name
      
    */
    /**************************************************************************/
    size_t max_name_length(std::vector<CountryInfo> const& v);

    /**************************************************************************/
    /*!
      \brief
         Sorts all elements in vector<CountryInfo> object referenced by rv using
         a sorting criterion specified by the comparison function pointed to by cmp
      
      \param rv
        A vector that contains unsorted data of all countries 

      \param cmp
        Comparison function pointer
    */
    /**************************************************************************/
    void sort(std::vector<CountryInfo>& rv, Ptr_Cmp_Func cmp);

    /**************************************************************************/
    /*!
      \brief
        Write the contents of the container referenced by v into the output stream
        referenced by os
      
      \param v
        A vector that contains sorted data of all countries 

      \param os
        Output stream
        
      \param fw
         The field width to be used when writing the country name
    */
    /**************************************************************************/
    void write_to_ostream(std::vector<CountryInfo> const& v,
                          std::ostream& os, size_t fw);

    /**************************************************************************/
    /*!
      \brief
        Compares the names of two countries lexicographically
      
      \param left
        Country 1

      \param right
        Country 2
        
      \return
        True if the country name referenced by left is lexicographically less
        than the country name referenced by right. Otherwise, the function
        returns false.
    */
    /**************************************************************************/
    bool cmp_name_less(CountryInfo const& left, CountryInfo const& right);

    /**************************************************************************/
    /*!
      \brief
        Compares the names of two countries lexicographically
      
      \param left
        Country 1

      \param right
        Country 2
        
      \return
        True if the country name referenced by left is lexicographically greater
        than the country name referenced by right. Otherwise, the function
        returns false.
    */
    /**************************************************************************/
    bool cmp_name_greater(CountryInfo const& left, CountryInfo const& right);

     /**************************************************************************/
    /*!
      \brief
        Compares the populations of two countries
      
      \param left
        Country 1

      \param right
        Country 2
        
      \return
        True if population of object referenced by left is numerically less than
        population of object referenced by right. Otherwise, the function
        returns false.
    */
    /**************************************************************************/
    bool cmp_pop_less(CountryInfo const& left, CountryInfo const& right);
    
    /**************************************************************************/
    /*!
      \brief
        Compares the populations of two countries
      
      \param left
        Country 1

      \param right
        Country 2
        
      \return
        True if population of object referenced by left is numerically greater than
        population of object referenced by right. Otherwise, the function
        returns false.
    */
    /**************************************************************************/
    bool cmp_pop_greater(CountryInfo const& left, CountryInfo const& right);
}

#endif