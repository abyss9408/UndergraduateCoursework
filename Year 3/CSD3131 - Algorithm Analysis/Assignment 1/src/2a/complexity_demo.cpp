#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <set>

// ==========================================
// COMPILATION INSTRUCTIONS
// ==========================================
/*
To compile and run:

g++ -std=c++17 -O2 -o complexity_demo complexity_demo.cpp
./complexity_demo

Expected output shows:
1. Linearithmic growth: time increases moderately as size doubles
2. Factorial growth: time explodes exponentially with each new element
*/

// ==========================================
// LINEARITHMIC O(n log n) ALGORITHM
// ==========================================

class DivideConquerMaxFinder
{
private:
    static long long operationCount;

public:
    /**
     * Find maximum element using divide and conquer approach.
     * This demonstrates O(n log n) behavior through recursive calls.
     * 
     * Time Complexity Analysis:
     * - Each level divides array in half: log n levels
     * - Each level processes all n elements: n work per level  
     * - Total: n * log n = O(n log n)
     * 
     * Note: This is intentionally inefficient for demonstration.
     * Finding max can be done in O(n), but this shows the pattern.
     */
    static int findMaxRecursive(const std::vector<int>& arr, int start, int end, int depth = 0)
    {
        // Base case
        if (start == end)
        {
            ++operationCount;// Count base case operation
            return arr[start];
        }
        
        // Divide
        int mid = (start + end) / 2;
        
        // Conquer - find max in both halves
        int leftMax = findMaxRecursive(arr, start, mid, depth + 1);
        int rightMax = findMaxRecursive(arr, mid + 1, end, depth + 1);
        
        // Combine - but here's where we add O(n log n) work
        // We scan the current range to verify our result (artificial work)
        int currentMax = std::max(leftMax, rightMax);
        ++operationCount; // Count the max comparison
        
        // This scan makes each recursive call do O(n) work at each of log n levels
        for (int i = start; i <= end; ++i)
        {
            ++operationCount; // Count each comparison in verification scan
            if (arr[i] > currentMax)
            {
                currentMax = arr[i];
            }
        }
        
        return currentMax;
    }
    
    static int findMax(const std::vector<int>& arr)
    {
        if (arr.empty())
        {
            throw std::invalid_argument("Array cannot be empty");
        }
        operationCount = 0; // Reset counter
        return findMaxRecursive(arr, 0, arr.size() - 1);
    }

    static long long getOperationCount()
    {
        return operationCount;
    }
    
    static void resetOperationCount()
    {
        operationCount = 0;
    }
};

// Initialize static member
long long DivideConquerMaxFinder::operationCount = 0;

// ==========================================
// FACTORIAL O(n!) ALGORITHM  
// ==========================================

class TeamFormationOptimizer
{
private:
    struct Formation
    {
        std::vector<std::string> players;
        double chemistryScore;
        
        Formation(const std::vector<std::string>& p, double score) 
            : players(p), chemistryScore(score) {}
    };
    
    /**
     * Calculate chemistry score for a team formation
     */
    static double calculateTeamChemistry(const std::vector<std::string>& formation)
    {
        double chemistry = 0.0;
        int n = formation.size();
        
        // Check chemistry between all pairs of players
        for (int i = 0; i < n; ++i)
        {
            for (int j = i + 1; j < n; ++j)
            {
                const std::string& player1 = formation[i];
                const std::string& player2 = formation[j];
                
                // Simple chemistry formula based on name compatibility
                chemistry += abs((int)player1.length() - (int)player2.length()) * 0.1;
                
                // Count common characters
                std::set<char> chars1(player1.begin(), player1.end());
                std::set<char> chars2(player2.begin(), player2.end());
                std::set<char> intersection;
                
                set_intersection(chars1.begin(), chars1.end(),
                               chars2.begin(), chars2.end(),
                               inserter(intersection, intersection.begin()));
                
                chemistry += intersection.size() * 0.3;
            }
        }
        
        return chemistry;
    }
    
    /**
     * Generate all possible permutations - this is the O(n!) part
     */
    static void generateAllPermutations(std::vector<std::string> items, int start, 
                                      std::vector<Formation>& allFormations)
                                      {
        if (start == items.size())
        {
            double score = calculateTeamChemistry(items);
            allFormations.emplace_back(items, score);
            return;
        }
        
        for (int i = start; i < items.size(); i++)
        {
            std::swap(items[start], items[i]);
            generateAllPermutations(items, start + 1, allFormations);
            std::swap(items[start], items[i]); // backtrack
        }
    }
    
public:
    /**
     * Find the optimal team formation by trying every possible arrangement
     * of players and evaluating each formation's 'chemistry score'.
     * 
     * This demonstrates O(n!) complexity because:
     * - We generate all possible permutations of players: n! permutations
     * - For each permutation, we calculate a score: O(n²) work per permutation
     * - Total: n! * n² = O(n! * n²) ≈ O(n!)
     */
    static std::vector<Formation> optimizeTeamFormation(const std::vector<std::string>& players)
    {
        if (players.empty())
        {
            return {};
        }
        
        std::vector<Formation> allFormations;
        std::vector<std::string> playersCopy = players;
        
        // Generate all n! permutations
        generateAllPermutations(playersCopy, 0, allFormations);
        
        // Sort by chemistry score (this adds O(n! log n!) but doesn't change overall complexity)
        std::sort(allFormations.begin(), allFormations.end(), 
             [](const Formation& a, const Formation& b)
             {
                return a.chemistryScore > b.chemistryScore;
             });
        
        return allFormations;
    }
};

// ==========================================
// DEMONSTRATION AND TIMING
// ==========================================

class ComplexityDemonstrator
{
public:
    static void demonstrateLinearithmic()
    {
        std::cout << "=== LINEARITHMIC O(n log n) DEMONSTRATION ===\n";
        std::cout << "Algorithm: Divide and Conquer Max Finder\n";
        std::cout << "(Artificially complex for demonstration)\n\n";
        
        std::vector<int> sizes = {2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1024000};
        
        for (int size : sizes)
        {
            // Create reverse sorted array for worst case
            std::vector<int> testArray(size);
            for (int i = 0; i < size; ++i)
            {
                testArray[i] = size - i;
            }
            
            auto start = std::chrono::high_resolution_clock::now();
            int result = DivideConquerMaxFinder::findMax(testArray);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            long long operations = DivideConquerMaxFinder::getOperationCount();

            std::cout << "Size: " << std::setw(3) << size 
                 << " | Time: " << std::setw(8) << duration.count() << "μs"
                 << " | Operations: " << std::setw(8) << operations
                 << " | Max found: " << result << '\n';
        }
        
        std::cout << '\n' << std::string(50, '=') << "\n\n";
    }
    
    static void demonstrateFactorial()
    {
        std::cout << "=== FACTORIAL O(n!) DEMONSTRATION ===\n";
        std::cout << "Algorithm: Team Formation Optimizer\n";
        std::cout << "(Tries every possible player arrangement)\n\n";
        
        std::vector<std::vector<std::string>> testCases =
        {
            {"Alice"},
            {"Alice", "Bob"},
            {"Alice", "Bob", "Charlie"}, 
            {"Alice", "Bob", "Charlie", "Diana"},
            {"Alice", "Bob", "Charlie", "Diana", "Eve"},
            {"Alice", "Bob", "Charlie", "Diana", "Eve", "Fred"},
            {"Alice", "Bob", "Charlie", "Diana", "Eve", "Fred", "Gwen"},
            {"Alice", "Bob", "Charlie", "Diana", "Eve", "Fred", "Gwen", "Howard"},
            {"Alice", "Bob", "Charlie", "Diana", "Eve", "Fred", "Gwen", "Howard", "Iris"},
            {"Alice", "Bob", "Charlie", "Diana", "Eve", "Fred", "Gwen", "Howard", "Iris", "John"}
        };
        
        for (const auto& players : testCases)
        {
            int n = players.size();
            long long expectedPermutations = 1;
            for (int i = 1; i <= n; ++i)
            {
                expectedPermutations *= i;
            }
            
            std::cout << "Testing with " << n << " players (" 
                 << expectedPermutations << " permutations to check):\n";
            
            auto start = std::chrono::high_resolution_clock::now();
            auto formations = TeamFormationOptimizer::optimizeTeamFormation(players);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            std::cout << "  Time taken: " << std::setw(10) << duration.count() << "μs\n";
            std::cout << "  Best formation: ";
            for (size_t i = 0; i < formations[0].players.size(); i++)
            {
                std::cout << formations[0].players[i];
                if (i < formations[0].players.size() - 1) std::cout << ", ";
            }
            std::cout << " (score: " << std::fixed << std::setprecision(2) 
                 << formations[0].chemistryScore << ")\n";
            std::cout << "  Permutations checked: " << formations.size() << "\n\n";
        }
    }
    
    static void demonstrateComplexities()
    {
        demonstrateLinearithmic();
        demonstrateFactorial();
    }
};

// ==========================================
// MAIN FUNCTION
// ==========================================

int main()
{
    try
    {
        ComplexityDemonstrator::demonstrateComplexities();
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
    
    return 0;
}