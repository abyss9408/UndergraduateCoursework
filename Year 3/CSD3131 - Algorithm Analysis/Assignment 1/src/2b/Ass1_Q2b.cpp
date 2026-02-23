// Ass1_Q2b.cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <ctime>

// Global counter to track operations
long long operation_count = 0;

// Function to calculate cross-border interference (the 'n' part of the work)
int process_borders(const std::vector<std::vector<int>>& grid, int x, int y, int size) {
    int border_score = 0;
    int mid = size / 2;
    // Process central column
    for (int i = 0; i < size; ++i) {
        border_score += grid[x + i][y + mid - 1] * grid[x + i][y + mid];
        operation_count++;
    }
    // Process central row
    for (int j = 0; j < size; ++j) {
        border_score += grid[x + mid - 1][y + j] * grid[x + mid][y + j];
        operation_count++;
    }
    return border_score;
}

// The recursive algorithm
int calculate_interference(const std::vector<std::vector<int>>& grid, int x, int y, int size) {
    operation_count++;
    // Base Case: If the grid is 1x1, there is no internal or border interference.
    if (size <= 1) {
        return 0;
    }

    int half_size = size / 2;

    // 1. Conquer: Recursive calls on the 4 sub-problems (4T(n/2))
    int tl_score = calculate_interference(grid, x, y, half_size);
    int tr_score = calculate_interference(grid, x, y + half_size, half_size);
    int bl_score = calculate_interference(grid, x + half_size, y, half_size);
    int br_score = calculate_interference(grid, x + half_size, y + half_size, half_size);

    // 2. Work: Process borders at the current level (O(n))
    int border_score = process_borders(grid, x, y, size);

    // 3. Combine: Sum the results
    return tl_score + tr_score + bl_score + br_score + border_score;
}

int main() {
    std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr)));
    std::uniform_int_distribution<int> distribution(1, 10);

    for (int n = 2; n <= 32; n *= 2) {
        // Create a grid and fill it with random values
        std::vector<std::vector<int>> grid(n, std::vector<int>(n));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                grid[i][j] = distribution(rng);
            }
        }
        
        operation_count = 0;
        int total_score = calculate_interference(grid, 0, 0, n);

        std::cout << "Grid size n = " << n << std::endl;
        std::cout << "Total Interference Score: " << total_score << std::endl;
        std::cout << "Total Operations: " << operation_count << std::endl;
        std::cout << "-------------------------" << std::endl;
    }
    return 0;
}