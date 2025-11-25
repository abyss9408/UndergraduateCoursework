#
# Compiler flags
#
CXX     = g++
CXX_FLAGS = -std=c++17 -pedantic-errors -Wall -Werror -Wextra
LDLIBS    = -lm

#
# Project files
#
SRCS = PRNG.cpp ocean.cpp ocean-driver2.cpp
OBJS = $(SRCS:.cpp=.o)
EXEC  = ocean2.out

#
# Debug build settings
#
DBGDIR = debug
DBGEXEC = $(DBGDIR)/$(EXEC)
DBGOBJS = $(addprefix $(DBGDIR)/, $(OBJS))
DBGCXX_FLAGS = -g -O0 -DDEBUG

#
# Release build settings
#
RELDIR = release
RELEXEC = $(RELDIR)/$(EXEC)
RELOBJS = $(addprefix $(RELDIR)/, $(OBJS))
RELCXX_FLAGS = -O3 -DNDEBUG

.PHONY: all clean debug-memory prep release remake

# Default build
all: prep release

#
# Debug rules
#
debug-memory: $(DBGEXEC)

$(DBGEXEC): $(DBGOBJS)
	$(CXX) $(CXX_FLAGS) $(DBGCXX_FLAGS) -o $(DBGEXEC) $^

$(DBGDIR)/%.o: %.cpp
	$(CXX) -c $(CXX_FLAGS) $(DBGCXX_FLAGS) -o $@ $<

#
# Release rules
#
release: $(RELEXEC)

$(RELEXEC): $(RELOBJS)
	$(CXX) $(CXX_FLAGS) $(RELCXX_FLAGS) -o $(RELEXEC) $^

$(RELDIR)/%.o: %.cpp
	$(CXX) -c $(CXX_FLAGS) $(RELCXX_FLAGS) -o $@ $<

#
# Other rules
#
prep:
	@mkdir -p $(DBGDIR) $(RELDIR)

remake: clean all

clean:
	rm -f $(RELEXEC) $(RELOBJS) $(DBGEXEC) $(DBGOBJS) $(DBGDIR)/*.txt $(RELDIR)/*.txt
	
.PHONY : test-all test1 test2 test3 test4 test5
test-all : test1 test2 test3 test4 test5

test1 : $(RELEXEC)
	./$(RELEXEC) 1 > ./$(RELDIR)/your-$@.txt
	diff -y --strip-trailing-cr --suppress-common-lines ./$(RELDIR)/your-$@.txt ./$@.txt

test2 : $(RELEXEC)
	./$(RELEXEC) 2 > ./$(RELDIR)/your-$@.txt
	diff -y --strip-trailing-cr --suppress-common-lines ./$(RELDIR)/your-$@.txt ./$@.txt

test3 : $(RELEXEC)
	./$(RELEXEC) 3 > ./$(RELDIR)/your-$@.txt
	diff -y --strip-trailing-cr --suppress-common-lines ./$(RELDIR)/your-$@.txt ./$@.txt

test4 : $(RELEXEC)
	./$(RELEXEC) 4 > ./$(RELDIR)/your-$@.txt
	diff -y --strip-trailing-cr --suppress-common-lines ./$(RELDIR)/your-$@.txt ./$@.txt

test5 : $(RELEXEC)
	./$(RELEXEC) 5 > ./$(RELDIR)/your-$@.txt
	diff -y --strip-trailing-cr --suppress-common-lines ./$(RELDIR)/your-$@.txt ./$@.txt
	
.PHONY : $(DBGDIR)-test-all $(DBGDIR)-test1 $(DBGDIR)-test2 $(DBGDIR)-test3 $(DBGDIR)-test4 $(DBGDIR)-test5 $(DBGDIR)-test6
$(DBGDIR)-test-all : $(DBGDIR)-test1 $(DBGDIR)-test2 $(DBGDIR)-test3 $(DBGDIR)-test4 $(DBGDIR)-test5

$(DBGDIR)-test1 : $(DBGEXEC)
	valgrind -s ./$(DBGEXEC) 1 > ./$(DBGDIR)/your-$@.txt
	diff -y --strip-trailing-cr --suppress-common-lines ./$(DBGDIR)/your-$@.txt ./test1.txt
	
$(DBGDIR)-test2 : $(DBGEXEC)
	valgrind -s ./$(DBGEXEC) 2 > ./$(DBGDIR)/your-$@.txt
	diff -y --strip-trailing-cr --suppress-common-lines ./$(DBGDIR)/your-$@.txt ./test2.txt
	
$(DBGDIR)-test3 : $(DBGEXEC)
	valgrind -s ./$(DBGEXEC) 3 > ./$(DBGDIR)/your-$@.txt
	diff -y --strip-trailing-cr --suppress-common-lines ./$(DBGDIR)/your-$@.txt ./test3.txt
	
$(DBGDIR)-test4 : $(DBGEXEC)
	valgrind -s ./$(DBGEXEC) 4 > ./$(DBGDIR)/your-$@.txt
	diff -y --strip-trailing-cr --suppress-common-lines ./$(DBGDIR)/your-$@.txt ./test4.txt
	
$(DBGDIR)-test5 : $(DBGEXEC)
	valgrind -s ./$(DBGEXEC) 5 > ./$(DBGDIR)/your-$@.txt
	diff -y --strip-trailing-cr --suppress-common-lines ./$(DBGDIR)/your-$@.txt ./test5.txt
