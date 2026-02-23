# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall

# Target executable
TARGET = 3a_sorting
SRC = 3a_sorting.cpp

# Default dataset size
SIZE ?= 10k

# Build the sorting program
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

# Run sorting program with selected dataset size
run: $(TARGET)
	./$(TARGET) $(SIZE)

# Clean executable and output files
clean:
	rm -f $(TARGET) counting_sort.txt selection_sort.txt
