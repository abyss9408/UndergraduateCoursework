// HLP2: Lab 3
// author: Prasanna Ghali
// Brief Description: 
// This file contains the main function for lab on problem-solving with strings.
// main takes input and output file names as command-line parameters.
// The words in input file are stored in a std::vector<std::string> container.
// Each word, in turn, is transformed to its Pig Latin version through external
// function hlp2::to_piglatin
// The English and translated Pig Latin words are written to an output file
// in a pretty format.

#include <iostream> // std::cout
#include <iomanip>  // std::setw, std::endl 
#include <string>   // std::string
#include <fstream>  // std::ifstream
#include <vector>   // std::vector<std::string>
#include <iterator> // std::istream_iterator<std::string>
#include "q.hpp"    // to_piglatin

int main(int argc, char *argv[]) {
  // sanity test to make sure program usage is correct
  if (argc < 3) {
    std::cout << "Usage: ./q.out input-text-file your-output-text-file\n";
    return 0;
  }

  // sanity test to make sure input file can be opened for reads
  std::ifstream ifs {argv[1]};
  if (!ifs) {
    std::cout << "Unable to open input file " << argv[1] << std::endl;
    return 0;
  }

  // open output file ...
  std::ofstream ofs {argv[2]};
  if (!ofs) {
    std::cout << "Unable to open output file " << argv[2] << std::endl;
    return 0;
  }

  std::istream_iterator<std::string> is{ifs}; // is points to first string in file
  std::istream_iterator<std::string> eos; // eos now points to end-of-file
  std::vector<std::string> words(is, eos); // fill vectors with strings from file
  ifs.close(); // return file resource back to system ...

  // for pretty printing, find largest length of words
  size_t max_wordlen {0};
  for (std::string const& w : words) {
    size_t len = w.length();
    max_wordlen = (len > max_wordlen) ? len : max_wordlen;
  }

  // do the actual work here ...
  for (std::string w : words) {
    size_t fill_width {(max_wordlen - w.length()) + 5UL};
    ofs << w << std::setw(fill_width) << " --> " << hlp2::to_piglatin(w) << "\n";
  }
}
