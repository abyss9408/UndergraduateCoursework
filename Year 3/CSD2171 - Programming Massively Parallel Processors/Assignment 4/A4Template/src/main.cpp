#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "helper.h"

constexpr int TOT_ITERS = 10;

int main(int argc, char* argv[]) {
    int n = 1000000;  // default: 1 million elements
    int version = 1;
    int baseline = 0;//change to  -1 for CPU;

    if (argc != 2) {
        printf("Usage: a4.exe <number-of-elements-to-sort>\n");
        return 1;
    }

    n = atoi(argv[1]);

    if (n < 0 || n > 20000000) {
        printf("n should be larger than 0 but less than 20000000\n\n");
        return 0;
    }
    //scanf("% d", &version);
    unsigned int* h_data = (unsigned int*)malloc(n * sizeof(unsigned int));
    unsigned int* h_test = (unsigned int*)malloc(n * sizeof(unsigned int));
    unsigned int* d_data;

    printf("\n===============================================\n");
    printf("RADIX SORT - All Operations on GPU\n");
    printf("===============================================\n");
    printf("Array size: %d elements\n", n);
    printf("Radix bits: %d (radix = %d)\n", RADIX_BITS, RADIX);
    printf("Section size: %d\n", SECTION_SIZE);
    printf("Coarsening factor: %d\n", COARSE_FACTOR);
    printf("Number of passes: %d\n\n", (32 + RADIX_BITS - 1) / RADIX_BITS);

    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(unsigned int)));

    // Generate random data properly
    srand(time(NULL));
    printf("Generating %d random integers...\n", n);
    for (int i = 0; i < n; i++) {
        // Generate 30-bit random numbers (0 to ~1 billion)
        h_data[i] = ((unsigned int)rand() << 15) | (unsigned int)rand();
        // Alternative: use C++11 <random> for better randomness
    }
    float ms = .0f;
    bool correct = false;
    float sum_latency_base = .0f;
    float sum_latency = .0f;
    for (int count = 0; count < TOT_ITERS; count++) {
        // ========= Test baseline===========
        printf("\n--- Version : baseline ---\n");

        // Copy test data
        memcpy(h_test, h_data, n * sizeof(unsigned int));
        CUDA_CHECK(cudaMemcpy(d_data, h_test, n * sizeof(unsigned int),
            cudaMemcpyHostToDevice));

        // Use timing wrapper function
        ms = time_radix_sort(d_data, n, baseline);

        // Copy back and verify
        CUDA_CHECK(cudaMemcpy(h_test, d_data, n * sizeof(unsigned int),
            cudaMemcpyDeviceToHost));

        correct = verify(h_test, n);
        sum_latency_base += ms;
        printf("Time: %.3f ms\n", ms);
        printf("Throughput: %.2f M keys/sec\n", (n / 1000000.0) / (ms / 1000.0));
        printf("Status: %s\n", correct ? "CORRECT" : "FAILED");

        // ========= Test submission ===========
        printf("\n--- Version %d ---\n", version);

        // Copy test data
        memcpy(h_test, h_data, n * sizeof(unsigned int));
        CUDA_CHECK(cudaMemcpy(d_data, h_test, n * sizeof(unsigned int),
            cudaMemcpyHostToDevice));

        // Use timing wrapper function
        ms = time_radix_sort(d_data, n, version);

        // Copy back and verify
        CUDA_CHECK(cudaMemcpy(h_test, d_data, n * sizeof(unsigned int),
            cudaMemcpyDeviceToHost));

        correct = verify(h_test, n);
        sum_latency += ms;
        printf("Time: %.3f ms\n", ms);
        printf("Throughput: %.2f M keys/sec\n", (n / 1000000.0) / (ms / 1000.0));
        printf("Status: %s\n", correct ? "CORRECT" : "FAILED");
    }

    sum_latency_base /= (float)(TOT_ITERS);
    sum_latency /= (float) (TOT_ITERS);

    printf("\nVersion %d Speedup (compared to baseline):%f (%f %f)\n",version, sum_latency_base / sum_latency, sum_latency_base, sum_latency );

    CUDA_CHECK(cudaFree(d_data));
    free(h_data);
    free(h_test);

    return 0;
}

