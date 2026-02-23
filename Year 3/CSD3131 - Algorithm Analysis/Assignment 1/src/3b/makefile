# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2

# Target executable
TARGET := 3b_radix

# Source files
SRCS := 3b.cpp
OBJS := $(SRCS:.cpp=.o)

# Default rule: build the program
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Run the program after building
run: $(TARGET)
	./$(TARGET)

run_extensive: $(TARGET)
	./$(TARGET) 1000000 32 256

test: CXXFLAGS += -D_TEST_
test: clean $(TARGET)
	./$(TARGET)

# Clean up build artifacts
clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all run clean
