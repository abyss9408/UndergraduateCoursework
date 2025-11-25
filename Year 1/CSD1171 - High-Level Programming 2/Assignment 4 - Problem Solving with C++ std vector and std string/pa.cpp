/**************************************************************************/
/*!
  \file pa.cpp

  \author Bryan Ang Wei Ze

  \par DP email: bryanweize.ang\@digipen.edu

  \par Course: CSD1171 High-Level Programming 2

  \par Programming Assignment #4

  \date 02-02-2024

  \brief
    This source file implements functions that read, process and transform an input
    file that has a table of names and populations of countries into four output
    files containing tables with countries names sorted in increasing and decreasing
    orders and population counts sorted in increasing and decreasing orders. 
    
*/
/**************************************************************************/
#include "pa.hpp"

namespace hlp2
{
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
    std::vector<CountryInfo> fill_vector_from_istream(std::istream& is)
    {
        std::vector<CountryInfo> all_countries;
        std::string line;
        std::string country_name;
        std::string population;
        long int pop;
        size_t country_name_beg{}, country_name_end{}, population_beg{}, population_end{};
        while (std::getline(is, line))
        {
            country_name_beg = line.find_first_of("ABCDEFGHIJKLMNOPQRSTUVWXYZ");
            country_name_end = line.find_last_of("abcdefghijklmnopqrstuvwxyz)");
            country_name = line.substr(country_name_beg, country_name_end-country_name_beg+1);
            population_beg = line.find_first_of("0123456789");
            population_end = line.find_last_of("0123456789");
            population = line.substr(population_beg, population_end-population_beg+1);

            // erase all commas from population string
            for (size_t i = 0; i < population.size(); ++i)
            {
                if (population.at(i) == ',')
                {
                    population.erase(population.begin() + i);
                }
            }

            pop = std::stol(population);
            CountryInfo country_data{country_name, pop};
            all_countries.push_back(country_data);
        }
        
        
        return all_countries;
    }
    
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
    size_t max_name_length(std::vector<CountryInfo> const& v)
    {
        size_t max_length{}, name_length{};
        for (CountryInfo const &country : v)
        {
            name_length = country.name.size();
            if (name_length > max_length)
            {
                max_length = name_length;
            }
        }
        
        return max_length;
    }
    
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
    void sort(std::vector<CountryInfo>& rv, Ptr_Cmp_Func cmp)
    {
        for (size_t i = 0; i < rv.size() - 1; ++i)
        {
            // assume first index as maximum or minimum
            int min_or_max = i;
            for (size_t j = i + 1; j < rv.size(); ++j)
            {
                /* making comparisons with other elements after
                currently selected element*/
                if (cmp(rv[j], rv[min_or_max]))
                {
                    // updating the maximum or minimum value
                    min_or_max = j;
                }
            }
            std::swap(rv[min_or_max], rv[i]);
        }
    }
    
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
                          std::ostream& os, size_t fw)
    {
        for (CountryInfo const &country : v)
        {
            os << std::setw(fw) << std::left << country.name << std::left << country.pop << '\n';
        }
    }
    
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
    bool cmp_name_less(CountryInfo const& left, CountryInfo const& right)
    {
        return (left.name < right.name) ? true : false;
    }
    
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
    bool cmp_name_greater(CountryInfo const& left, CountryInfo const& right)
    {
        return (left.name > right.name) ? true : false;
    }
    
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
    bool cmp_pop_less(CountryInfo const& left, CountryInfo const& right)
    {
        return (left.pop < right.pop) ? true : false;
    }
    
    
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
    bool cmp_pop_greater(CountryInfo const& left, CountryInfo const& right)
    {
        return (left.pop > right.pop) ? true : false;
    }
}