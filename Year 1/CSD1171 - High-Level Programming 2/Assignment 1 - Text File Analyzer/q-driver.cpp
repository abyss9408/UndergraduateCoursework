#include <iostream>
#include "q.hpp"

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cout << "Usage: ./q.out input-text-file input-text-file-stats\n";
    return 0;
  }

  // argv[1]: input text file to analyze
  // argv[2]: output text file containing analysis of input text file
  hlp2::q(argv[1], argv[2]);
}
