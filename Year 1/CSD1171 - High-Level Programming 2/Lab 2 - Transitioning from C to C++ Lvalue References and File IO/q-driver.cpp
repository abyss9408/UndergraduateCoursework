// HLP2
// Lab 2

#include <iostream>
#include "q.hpp"

/*
The program requires the name of a text file recording tsunami events
and the name of an output file in which to write a report of the tsunamis
and some summary statistics related to these tsunamis.
Without these text files, it is not possible for the program to do anything
useful. Therefore, it is important to ensure that the program receives
at least two command-line arguments in addition to the program's name.
*/
int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr <<  "Usage: ./q.out tsunami-data-file your-summary-output-file;\n";
    return 0;
  } 
  
  int num_events;
  hlp2::Tsunami *tsun_data = hlp2::read_tsunami_data(argv[1], num_events);
  if (tsun_data) {
    hlp2::print_tsunami_data(tsun_data, num_events, argv[2]);
    delete [] tsun_data;
  }
}
