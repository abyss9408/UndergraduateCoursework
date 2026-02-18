#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "helper.h"

// Step 1: Compute histogram for baseline kernel
/**
 * Computes per-block histograms for a specific digit (radix) position
 *
 * @param d_in          Input array of unsigned integers (device memory)
 * @param d_block_hist  Output histogram per block (device memory)
 *                      Size: [num_blocks × RADIX] for byte-wise radix (8-bit)
 * @param shift         Number of bits to shift right to extract desired digit
 *                      (0, 8, 16, 24 for 4-pass radix sort on 32-bit ints)
 * @param n             Total number of elements in input array
 *
 * Kernel Launch Configuration:
 *   - Grid:  (num_blocks, 1, 1) where num_blocks = (n+section_size-1) / section_size
 *   - Block: (section_size, 1, 1) typically 256 or 512 threads
 */
__global__ void compute_histogram_base(unsigned int* d_in,
    unsigned int* d_hist,
    unsigned int shift,
    unsigned int n) {

}

// Step 2: GPU Global Prefix Sum for baseline kernel
/**
 * Computes global offsets for each block and each digit using prefix sum
 *
 * @param d_hist           Per-block histograms from step 1 (device memory)
 *                         Size: [num_blocks × RADIX]
 * @param d_block_offsets  Output: Starting offset for each block in final array
 *                         Size: [num_blocks] (or [num_blocks + 1] for exclusive scan)
 * @param d_digit_offsets  Output: Starting offset for each digit value (0-(RADIX-1))
 *                         within the global array (device memory)
 *                         Size: [RADIX] (or [RADIX+1] for exclusive scan)
 * @param num_blocks       Number of blocks used in histogram computation
 *
 * Kernel Launch Configuration:
 *   - Single thread or small grid that performs parallel prefix sum
 */
__global__ void compute_global_offsets_base(unsigned int* d_hist,
    unsigned int* d_block_offsets,
    unsigned int* d_digit_offsets,
    int num_blocks) {
}


// Step 3: Scatter kernel for baseline
/**
 * Scatters elements to their sorted positions based on digit values
 *
 * @param d_out            Output sorted array (device memory)
 * @param d_in             Input array to be sorted (device memory)
 * @param d_block_offsets  Starting offset for each block (from step 2)
 *                         Size: [num_blocks]
 * @param d_digit_offsets  Running offsets for each digit value (from step 2)
 *                         Size: [RADIX] (updated atomically during scatter)
 * @param shift            Number of bits to shift to extract current digit
 * @param n                Total number of elements
 *
 * Kernel Launch Configuration:
 *   - Grid:  (num_blocks, 1, 1) where num_blocks = (n+section_size-1) / section_size
 *   - Block: (section_size, 1, 1) same as histogram kernel
 */
__global__ void scatter_base(unsigned int* d_out,
    unsigned int* d_in,
    unsigned int* d_block_offsets,
    unsigned int* d_digit_offsets,
    unsigned int shift,
    unsigned int n) {

}

/**
 * Host function that orchestrates one iteration of radix sort for specific digit(s)
 *
 * @param d_out            Output buffer for this iteration (device memory)
 * @param d_in             Input buffer for this iteration (device memory)
 * @param d_hist           Temporary buffer for per-block histograms
 *                         Size: num_blocks × RADIX × sizeof(unsigned int)
 * @param d_block_offsets  Temporary buffer for block offsets
 *                         Size: num_blocks × sizeof(unsigned int)
 * @param d_digit_offsets  Temporary buffer for digit offsets
 *                         Size: RADIX × sizeof(unsigned int)
 * @param shift            Current bit shift (0, 8, 16, 24 for 32-bit ints)
 * @param n                Total number of elements to sort
 *
 * Memory Requirements:
 *   - d_hist:          num_blocks * RADIX * sizeof(unsigned int)
 *   - d_block_offsets: num_blocks * sizeof(unsigned int)
 *   - d_digit_offsets: RADIX * sizeof(unsigned int)
 *
 * Note: This function handles launching all three kernels in sequence
 */
extern "C" void radix_sort_iteration_base(unsigned int* d_out,
    unsigned int* d_in,
    unsigned int* d_hist,
    unsigned int* d_block_offsets,
    unsigned int* d_digit_offsets,
    unsigned int shift,
    unsigned int n) {

    // Calculate grid size
    int items_per_block = SECTION_SIZE;
    int num_blocks = (n + items_per_block - 1) / items_per_block;
    num_blocks = min(num_blocks, 65535);  // Safety limit

    // Step 1: Compute histogram
    compute_histogram_base <<<num_blocks, SECTION_SIZE >>> (d_in, d_hist, shift, n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 2: GPU prefix sum
    compute_global_offsets_base <<<1, RADIX >>> (d_hist, d_block_offsets, d_digit_offsets, num_blocks);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 3: Scatter with selected kernel
    scatter_base <<<num_blocks, SECTION_SIZE >>>
            (d_out, d_in, d_block_offsets, d_digit_offsets, shift, n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

