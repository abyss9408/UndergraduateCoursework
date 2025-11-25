/*
author: Prasanna Ghali
Brief Description: 
This driver program exercises the functions declared in interface file pa.h.
The aim of this lab is to expose students to the following concepts:
1) input/output streams presented in C++ standard library
2) string and vector types presented in C++ standard library. After completing
this lab, you will have experience in using objects of vector and string type
to solve small problems.
3) Using function pointers
4) Implementing your own sorting algorithm with a call-back mechanism.

If you are not refactoring the makefile from previous Labs, then compile
and link using:
g++ -std=c++17 -pedantic-errors -Wall -Wextra -Werror pa-driver.cpp pa.cpp -o pa.out

Use the command-line to specify the text file containing information about world
populations. If the data is in text file "worldpop.txt" the command is:
./pa.out worldpop.txt

The application will generate the following files:
a) your-name-asc.txt that writes country information in *ascending* order of country name
b) your-name-des.txt that writes country information in *descending* order of country name
c) your-pop-asc.txt that writes country information in *ascending* order of country population
d) your-pop-des.txt that writes country information in *descending* order of country population
******************************************************************************/

#include "pa.hpp"
#include <fstream>
#include <array>

// main receives one parameter from the command-line providing the name of
// the input file to read population data from.
int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: pa.exe input-text-file" << std::endl;
    return 0;
  }

  // open input file - perform sanity check to ensure file is present ...
  std::ifstream ifs {argv[1]};
  if (!ifs) { // file opened? NO - then abort program
    std::cerr << "Cannot open input file \"" << argv[1] << "\"" << std::endl;
    return 0;
  }

  // if the input file name is worldpop.txt, the output file name is determined as:
  // prefix (of input file): worldpop + out_suffix_fn[i] where i is between 0 and 3
  size_t const num_out_files {4UL};
  std::array<std::string, num_out_files> const out_fn { // specify output file names
    "your-name-asc.txt", "your-name-des.txt",
    "your-pop-asc.txt",  "your-pop-des.txt"
  };

  // define an array of function pointers to store four function pointers - each
  // pointing to a particular way in which the CountryInfo can be sorted
  std::array<HLP2::Ptr_Cmp_Func, num_out_files> const cmp_fos {
    HLP2::cmp_name_less, HLP2::cmp_name_greater,
    HLP2::cmp_pop_less,  HLP2::cmp_pop_greater
  };

  // see pa.hpp for implementation details and hints ...
  // you must implement this!!!
  std::vector<HLP2::CountryInfo> ci = HLP2::fill_vector_from_istream(ifs); 
  ifs.close(); // make sure to return file handle back to system ...

  // get length of longest name in vector container ...
  size_t longest_name = HLP2::max_name_length(ci); // you must implement this!!!

  // write output files
  for (size_t i = 0; i < num_out_files; ++i) {
    // sort the container using the comparison function cmp_fos[i]
    HLP2::sort(ci, cmp_fos[i]); // you must implement this!!!

    // open output file stream ...
    std::ofstream ofs {out_fn[i]};
    if (!ofs) { // file opened? NO - then abort program
      std::cerr << "Cannot open output file: " << out_fn[i] << std::endl;
      return 0;
    }

    // write sorted container to text file ...
    HLP2::write_to_ostream(ci, ofs, longest_name+3); // you must implement this!!!
    ofs.close();
  }
}
