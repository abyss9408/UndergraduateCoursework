# name of C++ compiler
CXX       = g++
# options to C++ compiler
CXX_FLAGS = -std=c++17 -pedantic-errors -Wall -Wextra -Werror
# flag to linker to make it link with math library
LDLIBS    = -lm
# list of object files
OBJS      = vpl_driver.o shape.o ellipse.o polygon.o process.o
# name of executable program
EXEC      = vpl_execution.out

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
# text $(EXEC) will be replaced with value sllist.out
# text $(OBJS) will be substituted with list of prerequisites in line 10
# line 31 says to build $(EXEC) using command $(CXX) with
# options $(CXX_FLAGS) specified on line 6
$(EXEC) : $(OBJS)
	$(CXX) $(CXX_FLAGS) $(OBJS) -o $(EXEC) $(LDLIBS)

shape.o : shape.cpp shape.hpp
	$(CXX) $(CXX_FLAGS) -c shape.cpp -o shape.o

polygon.o : polygon.cpp polygon.hpp
	$(CXX) $(CXX_FLAGS) -c polygon.cpp -o polygon.o

ellipse.o : ellipse.cpp ellipse.hpp
	$(CXX) $(CXX_FLAGS) -c ellipse.cpp -o ellipse.o

process.o : process.cpp process.hpp
	$(CXX) $(CXX_FLAGS) -c process.cpp -o process.o

vpl_driver.o : vpl_driver.cpp shape.hpp polygon.hpp ellipse.hpp process.hpp
	$(CXX) $(CXX_FLAGS) -c vpl_driver.cpp -o vpl_driver.o

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

.PHONY : test_ellipse
test_ellipse : $(EXEC)
	./$(EXEC) 1 > your-output-ellipse.txt
	diff -y --strip-trailing-cr --suppress-common-lines your-output-ellipse.txt output-ellipse.txt

.PHONY : test_polygon
test_polygon : $(EXEC)
	./$(EXEC) 2 > your-output-polygon.txt
	diff -y --strip-trailing-cr --suppress-common-lines your-output-polygon.txt output-polygon.txt

.PHONY : test
test : $(EXEC)
	./$(EXEC) 3 > your-output-all.txt
	diff -y --strip-trailing-cr --suppress-common-lines your-output-all.txt output-all.txt

.PHONY : leak
leak : $(EXEC)
	valgrind --leak-check=full ./$(EXEC) 3