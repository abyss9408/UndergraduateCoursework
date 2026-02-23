#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

// Utility function: get maximum value in array
int getMax(const std::vector<int>& arr) {
    return *std::max_element(arr.begin(), arr.end());
}

// Helper for generating arrays
std::vector<int> getRandArray(int count, int bitlen){
    std::vector<int> data;
    data.reserve(count);

    // Use a random number generator
    std::mt19937 rng(std::random_device{}());  // Mersenne Twister engine
    std::uniform_int_distribution<int> dist(0, (0x1<<bitlen)-1); // range [0, bits]

    // Generate N random integers
    for (int i = 0; i < count; i++) {
        data.push_back(dist(rng));
    }
    return data;
}

// Helper to print arrays
void printArray(std::vector<int> const& arr, std::string const& arr_name){
    std::cout << arr_name << ":\n";
    for (int num : arr) std::cout << num << " ";
    std::cout << "\n";
}

bool testCorrectness(std::vector<int> const& arr){
    for(long unsigned int i{}, j{1}; j < arr.size(); i++, j++){
        if (arr[i] > arr[j]){
            return false;
        }
    }
    return true;
}

// Stable counting sort used by radix sort
void countingSort(std::vector<int>& arr, int exp, int base) {
    int n = arr.size();
    std::vector<int> output(n);   // output array
    std::vector<int> count(base, 0); // base digits

    // Count occurrences of each digit
    for (int i = 0; i < n; i++) {
        int digit = (arr[i] / exp) % base;
        count[digit]++;
    }

    // Convert count[i] to position indices
    for (int i = 1; i < base; i++) {
        count[i] += count[i - 1];
    }

    // Build output array (iterate backwards for stability)
    for (int i = n - 1; i >= 0; i--) {
        int digit = (arr[i] / exp) % base;
        output[count[digit] - 1] = arr[i];
        count[digit]--;
    }

    // Copy back to arr
    for (int i = 0; i < n; i++) {
        arr[i] = output[i];
    }
}

// Main radix sort function
void radixSort(std::vector<int>& arr, int base) {
    int maxVal = getMax(arr);

    // Apply counting sort for each digit (exp = 1, base, base^2, ...)
    for (int exp = 1; maxVal / exp > 0; exp *= base) {
        countingSort(arr, exp, base);
    }
}

void radixSortBenchmark(int count, int bitlen, int radixbase){
    int genCount = count;
    int genBitLen = bitlen;
    int radixBase = radixbase;
    genBitLen > 64 ? genBitLen=64:genBitLen;
    int digits = log(0x1ull<<genBitLen)/ log(radixBase);
    digits > 0 ? digits: digits = 1;
    
    std::vector<int> data_radix = getRandArray(genCount, genBitLen);
    std::vector<int> data_std = data_radix;

    // radix sort
    auto start = std::chrono::high_resolution_clock::now();
    radixSort(data_radix, radixBase);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> radix_time = end - start;

    // std::sort (O(n log n))
    start = std::chrono::high_resolution_clock::now();
    std::sort(data_std.begin(), data_std.end());
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double,std::micro> std_time = end - start;

    // Verify correctness
    bool correct = (data_radix == data_std);

    std::cout << "Sorted correctly: " << (correct ? "True" : "False") << "\n";
    std::cout << "K bit length value: " << genBitLen << "\n";
    std::cout << "Radix Base: " << radixBase << "\n";
    std::cout << "Radix Digits: " << digits << "\n";
    std::cout << "Radix Sort time: " << radix_time.count() << " μs\n";
    std::cout << "std::sort time:  " << std_time.count() << " μs\n";
}

int main(int argc, char* argv[]) {
#ifndef _TEST_
    int genCount = argc > 1 ? std::stoi(argv[1]) : 100;
    int genBitLen = argc > 2 ? std::stoi(argv[2]) : 8;
    int radixBase = argc > 3 ? std::stoi(argv[3]) : 16;
    genBitLen > 32 ? genBitLen=32:genBitLen;
    int digits = log(0x1ull<<genBitLen)/ log(radixBase);
    digits > 0 ? digits: digits = 1;
    
    std::vector<int> data_radix = getRandArray(genCount, genBitLen);
    std::vector<int> data_std = data_radix;

    // radix sort
    auto start = std::chrono::high_resolution_clock::now();
    radixSort(data_radix, radixBase);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> radix_time = end - start;

    // std::sort (O(n log n))
    start = std::chrono::high_resolution_clock::now();
    std::sort(data_std.begin(), data_std.end());
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> std_time = end - start;

    // Verify correctness
    bool correct = (data_radix == data_std);

    std::cout << "Sorted correctly: " << (correct ? "True" : "False") << "\n";
    std::cout << "K bit length value: " << genBitLen << "\n";
    std::cout << "Radix Base: " << radixBase << "\n";
    std::cout << "Radix Digits: " << digits << "\n";
    std::cout << "Radix Sort time: " << radix_time.count() << " seconds\n";
    std::cout << "std::sort time:  " << std_time.count() << " seconds\n";
#else
    std::cout << "Test 1. 8 bit integer, 1000 elements, base 4\n";
    radixSortBenchmark(1000, 8, 4);
    std::cout << "\nTest 2. 16 bit integer, 10000 elements, base 4\n";
    radixSortBenchmark(10000, 16, 4);
    std::cout << "\nTest 3. 32 bit integer, 100000 elements, base 16\n";
    radixSortBenchmark(100000, 32, 16);
    std::cout << "\nTest 4. 32 bit integer, 1000000 elements, base 256\n";
    radixSortBenchmark(1000000, 32, 256);
#endif

    return 0;
}
