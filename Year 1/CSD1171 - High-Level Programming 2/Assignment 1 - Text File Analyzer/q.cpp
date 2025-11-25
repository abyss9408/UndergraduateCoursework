/*!************************************************************************
\file q.cpp
\author Bryan Ang Wei Ze
\par DP email: bryanweize.ang\@digipen.edu
\par Course: CSD1171 High-Level Programming 2
\par Programming Assignment #1
\date 01-12-2024
\brief
This source file implements a function inside namespace hlp2 that analyze a
file and output an analysis file about the statistics of the input file
**************************************************************************/
#include <iostream>
#include <iomanip>
#include <fstream>
#define NO_OF_ALPHABETS 26

/*!************************************************************************
\namespace hlp2
**************************************************************************/
namespace hlp2 {
/*!***********************************************************************
\brief
Implementation of function that computes total number of characters, number of
letters, white spaces, digits, characters, respective letters, integers,
sum of integers and average of integers
\param[in] input_filename
The file to be analyzed.
\param[in] analysis_file
The name of the analysis file that will be created based on the computed 
statistics of the input file.
**************************************************************************/
  void q(char const *input_filename, char const *analysis_file)
  {
    char read_char{'\0'};
    char string[50]{'\0'};
    int read_int{0}, integer_sum{0};
    double integer_avg{0.0};
    int char_count{0}, letter_count{0}, white_space_count{0}, digit_count{0}, other_count{0},
    uppercase_count{0}, lowercase_count{0}, res_letter_count[26]{0}, integer_count{0};


    std::fstream ifs(input_filename,std::ios_base::in);
    std::fstream ofs(analysis_file, std::ios_base::out);
    if (!ifs.is_open())
    {
      std::cout << "File "<<input_filename<<" not found."<<'\n';
    }
    
    // count all characters
    while (ifs.get(read_char))
    {
      char_count++;
      
      if (islower(read_char) || isupper(read_char))
      {
        letter_count++;
        if (islower(read_char))
        {
          lowercase_count++;
        }

        for (int i = 0; i < NO_OF_ALPHABETS; i++)
        {
          if (read_char == 'a'+i || read_char == 'A'+i)
          {
            res_letter_count[i]++;
            break;
          }
        }
      }
      else if (read_char == ' ' || read_char == '\n')
      {
        white_space_count++;
      }
      else if (isdigit(read_char))
      {
        digit_count++;
        
      }
      else
      {
        other_count++;
      }
    }

    ifs.clear();
    ifs.seekg(std::ios_base::beg);

    // count integers
    while (ifs.get(read_char)) 
    {
      if (isdigit(read_char))
      {
        integer_count++;
            char next_char = ifs.peek();
            while (isdigit(next_char))
              {
                ifs.get(next_char);
                next_char = ifs.peek();
              }
      }
    }
    
    ifs.clear();
    ifs.seekg(std::ios_base::beg);

    // read text file contents as string, extract integers and compute the sum
    while (ifs >> string)
    {
      for (int i = 0; string[i] != '\0'; i++)
      {
        if (isdigit(string[i]))
        {
          read_int = 10 * read_int + (string[i] - '0');
        }
        else
        {
          integer_sum+= read_int;
          read_int=0;
        }
      }
      
    }
    
    integer_sum+=read_int;
    uppercase_count = letter_count-lowercase_count;
    if (integer_count)
    {
      integer_avg = static_cast<double>(integer_sum)/integer_count;
    }


    ifs.close();
    
    
    // output statistics of file to analysis file
    ofs<< "Statistics for file: "<<input_filename<<'\n';
    ofs<<"---------------------------------------------------------------------"<<'\n';
    ofs<<'\n';
    ofs<< "Total # of characters in file: "<<char_count<<'\n';
    ofs<<'\n';
    ofs<< "Category            How many in file             % of file"<<'\n';
    ofs<<"---------------------------------------------------------------------"<<'\n';
    
    ofs<< std::fixed;
    ofs<< std::setprecision(2);

    ofs<< "Letters"<<std::setw(29)<<letter_count<< 
    std::setw(20)<<static_cast<double>(letter_count)/char_count*100<<" %\n";

    ofs<< "White space"<<std::setw(25)<<white_space_count<<
    std::setw(20)<<static_cast<double>(white_space_count)/char_count*100<<" %\n";

    ofs<< "Digits"<<std::setw(30)<<digit_count<<
    std::setw(20)<<static_cast<double>(digit_count)/char_count*100<<" %\n";

    ofs<< "Other characters"<<std::setw(20)<<other_count<<
    std::setw(20)<<static_cast<double>(other_count)/char_count*100<<" %\n";

    ofs<<'\n';
    ofs<<'\n';
    ofs<<"LETTER STATISTICS"<<'\n';
    ofs<<'\n';
    ofs<<"Category            How many in file      % of all letters"<<'\n';
    ofs<<"---------------------------------------------------------------------"<<'\n';

    ofs<<"Uppercase"<<std::setw(27)<<uppercase_count<<
    std::setw(20)<<static_cast<double>(uppercase_count)/letter_count*100<<" %\n";

    ofs<<"Lowercase"<<std::setw(27)<<lowercase_count<<
    std::setw(20)<<static_cast<double>(lowercase_count)/letter_count*100<<" %\n";

    for (int i = 0; i < NO_OF_ALPHABETS; i++)
    {
      char letter = 'a';
      ofs<< static_cast<char>(letter+i) <<std::setw(35)<<
      res_letter_count[i]<< std::setw(20) << static_cast<double>(res_letter_count[i])/letter_count*100<<" %\n";
    }
    
    ofs<<'\n';
    ofs<<'\n';
    ofs<<"NUMBER ANALYSIS"<<'\n';
    ofs<<'\n';
    ofs<<"Number of integers in file:          "<<integer_count<<'\n';
    ofs<<"Sum of integers:                     "<<integer_sum<<'\n';
    ofs<<"Average of integers:                 "<<integer_avg<<'\n';
    ofs<<"_____________________________________________________________________"<<'\n';
    ofs.close();
  }
}
