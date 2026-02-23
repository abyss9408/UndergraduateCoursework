How to build the program:
    1. ensure the computer has make
    2. run the command: make
    3. the build program will be stored as 3b_radix

How to run the program:
    1. run the command: ./3b_radix {number of elements} {bit length} {base}
    Explanation of parameters:
        {number of element} : set the number of elements to generate a randomised array
        {bit length}        : set the highest bit integer value that the generator can generate
        {base}              : set the base used in the radix sort. (default is base16 hexadecimal)

How to run the test suite:
    1. run the comman: make test

Limitations:
    1. the maximum number of elements cannot be greater than 2^32
    2. the maximum bit length value is 32
    3. the maximum base value is 2^32