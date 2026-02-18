#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "helper.h"

// ============================================================================
// HOST FUNCTIONS
// ============================================================================

float time_radix_sort(unsigned int* d_data,
    unsigned int n,
    int kernel_version) {
    unsigned int* d_temp, * d_hist, * d_block_offsets, * d_digit_offsets;
    void (*radix_sort_iter_func_ptr)(unsigned int*,
        unsigned int*,
        unsigned int*,
        unsigned int*,
        unsigned int*,
        unsigned int,
        unsigned int);
    if (kernel_version == 0)
        radix_sort_iter_func_ptr = radix_sort_iteration_base;
    else
        radix_sort_iter_func_ptr = radix_sort_iteration;
    // 1. Calculate grid size requirements
    int items_per_block = SECTION_SIZE;
    int num_blocks = (n + items_per_block - 1) / items_per_block;

    // 2. ALLOCATE ONCE (Outside the loop)
    CUDA_CHECK(cudaMalloc(&d_temp, n * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_hist, RADIX * num_blocks * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_block_offsets, RADIX * num_blocks * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_digit_offsets, RADIX * sizeof(unsigned int)));
    cudaEvent_t start, stop;
    float ms = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    unsigned int* d_in = d_data, * d_out = d_temp;
    int num_passes = (32 + RADIX_BITS - 1) / RADIX_BITS;

    for (int pass = 0; pass < num_passes; pass++) {
        unsigned int shift = pass * RADIX_BITS;

        // USE THE FULLY GPU ITERATION HERE
        radix_sort_iter_func_ptr(d_out, d_in, d_hist, d_block_offsets, d_digit_offsets, shift, n);

        // Swap pointers for the next pass
        unsigned int* temp = d_in; d_in = d_out; d_out = temp;
    }

    // Copy back final result if necessary
    if (d_in != d_data) {
        CUDA_CHECK(cudaMemcpy(d_data, d_in, n * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    // 3. FREE ONCE
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_hist));
    CUDA_CHECK(cudaFree(d_block_offsets));
    CUDA_CHECK(cudaFree(d_digit_offsets));

    return ms;
}

bool verify(unsigned int* data, int n) {
    for (int i = 1; i < n; i++) {
        if (data[i] < data[i - 1]) {
            printf("Error at position %d: %u > %u\n", i, data[i - 1], data[i]);
            return false;
        }
    }
    return true;
}
