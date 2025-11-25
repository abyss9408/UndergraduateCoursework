# name of C++ compiler
CXX       = g++
# options to C++ compiler
CXX_FLAGS = -std=c++17 -pedantic-errors -Wall -Wextra -Werror
# flag to linker to make it link with math library
LDLIBS    = -lm
# list of object files
OBJS      = ocean-driver.o ocean.o PRNG.o
# name of executable program
EXEC      = ocean.out

# by convention the default target (the target that is built when writing
# only make on the command line) should be called all and it should
# be the first target in a makefile
all : $(EXEC)

# however, the problem that arises with the previous rule is that make
# will think all is the name of the target file that should be created
# so, we tell make that all is not a file name
.PHONY : all

# this rule says that target $(EXEC) will be built if prerequisite
# files $(OBJS) have changed more recently than $(EXEC)
# text $(EXEC) will be replaced with value ocean.out
# text $(OBJS) will be substituted with list of prerequisites in line 10
# line 31 says to build $(EXEC) using command $(CXX) with
# options $(CXX_FLAGS) specified on line 6
$(EXEC) : $(OBJS)
	$(CXX) $(CXX_FLAGS) $(OBJS) -o $(EXEC) $(LDLIBS)

# target ocean.o depends on ocean.cpp, ocean.h
# and is created with command $(CXX) given the options $(CXX_FLAGS)
ocean.o : ocean.cpp ocean.h
	$(CXX) $(CXX_FLAGS) -c ocean.cpp -o ocean.o
	
# target driver-sample.o depends on both driver-sample.cpp, ocean.h, and prng.h
# and is created with command $(CXX) given the options $(CXX_FLAGS)
ocean-driver.o : ocean-driver.cpp ocean.h PRNG.h
	$(CXX) $(CXX_FLAGS) -c ocean-driver.cpp -o ocean-driver.o

PRNG.o : PRNG.cpp PRNG.h
	$(CXX) $(CXX_FLAGS) -c PRNG.cpp -o PRNG.o

# says that clean is not the name of a target file but simply the name for
# a recipe to be executed when an explicit request is made
.PHONY : clean
# clean is a target with no prerequisites;
# typing the command in the shell: make clean
# will only execute the command which is to delete the object files
clean :
	rm -f $(OBJS) $(EXEC)

# says that rebuild is not the name of a target file but simply the name
# for a recipe to be executed when an explicit request is made
.PHONY : rebuild
# rebuild is for starting over by removing cleaning up previous builds
# and starting a new one
rebuild :
	$(MAKE) clean
	$(MAKE)

.PHONY : test-all test1 test2 test3 test4 test5 test6
test-all : test1 test2 test3 test4 test5 test6

test1 : $(EXEC)
	./$(EXEC) 1 > your-$@.txt
	diff -y --strip-trailing-cr --suppress-common-lines your-$@.txt $@.txt

test2 : $(EXEC)
	./$(EXEC) 2 > your-$@.txt
	diff -y --strip-trailing-cr --suppress-common-lines your-$@.txt $@.txt

test3 : $(EXEC)
	./$(EXEC) 3 > your-$@.txt
	diff -y --strip-trailing-cr --suppress-common-lines your-$@.txt $@.txt

test4 : $(EXEC)
	./$(EXEC) 4 > your-$@.txt
	diff -y --strip-trailing-cr --suppress-common-lines your-$@.txt $@.txt

test5 : $(EXEC)
	./$(EXEC) 5 > your-$@.txt
	diff -y --strip-trailing-cr --suppress-common-lines your-$@.txt $@.txt

test6 : $(EXEC)
	./$(EXEC) 6 > your-$@.txt
	diff -y --strip-trailing-cr --suppress-common-lines your-$@.txt $@.txt
