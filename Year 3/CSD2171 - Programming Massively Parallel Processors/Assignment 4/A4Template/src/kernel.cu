#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "helper.h"

// Step 1: Compute block histograms 
/**
 * Computes per-block histograms for a specific digit (radix) position
 *
 * @param d_in          Input array of unsigned integers (device memory)
 * @param d_block_hist  Output histogram per block (device memory)
 *                      Size: [RADIX × num_blocks], digit-major layout:
 *                      d_hist[digit * num_blocks + block]
 * @param shift         Number of bits to shift right to extract desired digit
 *                      (0, 8, 16, 24 for 4-pass radix sort on 32-bit ints)
 * @param n             Total number of elements in input array
 *
 * Kernel Launch Configuration:
 *   - Grid:  (num_blocks, 1, 1) where num_blocks = (n+section_size-1) / section_size
 *   - Block: (section_size, 1, 1) typically 256 or 512 threads
 */
__global__ void compute_histogram(
    unsigned int* d_in,
    unsigned int* d_block_hist,
    unsigned int shift,
    unsigned int n)
{
    __shared__ unsigned int s_hist[RADIX];

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Initialize shared histogram to 0
    if (tid < RADIX)
        s_hist[tid] = 0;
    __syncthreads();

    // Each thread processes COARSE_FACTOR elements (thread coarsening).
    // RADIX=16 bins occupy 16 distinct shared-memory banks → zero bank conflicts.
    int base_idx = bid * SECTION_SIZE * COARSE_FACTOR + tid;
    for (int i = 0; i < COARSE_FACTOR; i++) {
        int idx = base_idx + i * SECTION_SIZE;
        if (idx < (int)n) {
            unsigned int digit = (d_in[idx] >> shift) & (RADIX - 1);
            atomicAdd(&s_hist[digit], 1);
        }
    }
    __syncthreads();

    // Write to global memory in digit-major layout: d_hist[digit * num_blocks + block]
    if (tid < RADIX)
        d_block_hist[tid * gridDim.x + bid] = s_hist[tid];
}

// Step 2: GPU Global Prefix Sum 
/**
 * Computes global offsets for each block and each digit using prefix sum
 *
 * @param d_hist           Per-block histograms from step 1 (device memory)
 *                         Size: [RADIX × num_blocks], digit-major layout
 * @param d_block_offsets  Output: Starting offset for each (digit, block) pair
 *                         Size: [RADIX × num_blocks], digit-major layout
 * @param d_digit_offsets  Output: Starting offset for each digit value (0-(RADIX-1))
 *                         within the global array (device memory)
 *                         Size: [RADIX]
 * @param num_blocks       Number of blocks used in histogram computation
 *
 * Kernel Launch Configuration:
 *   - Single thread or small grid that performs parallel prefix sum
 */
__global__ void compute_global_offsets(unsigned int* d_hist,
    unsigned int* d_block_offsets,
    unsigned int* d_digit_offsets,
    int num_blocks)
{
    // Each thread handles one digit (tid = 0 to RADIX-1)
    int digit = threadIdx.x;

    if (digit >= RADIX) return;

    // Compute prefix sum for this digit across all blocks (digit-major layout)
    // This gives us the starting offset for each block's contribution to this digit
    unsigned int sum = 0;
    for (int block = 0; block < num_blocks; block++) {
        unsigned int count = d_hist[digit * num_blocks + block];
        // Store exclusive prefix sum (offset before adding this block's count)
        d_block_offsets[digit * num_blocks + block] = sum;
        sum += count;
    }

    // Store total count for this digit (will be used for computing digit offsets)
    __shared__ unsigned int s_digit_totals[RADIX];
    s_digit_totals[digit] = sum;
    __syncthreads();

    // Compute exclusive prefix sum across digits to get starting position for each digit
    if (digit == 0) {
        unsigned int offset = 0;
        for (int d = 0; d < RADIX; d++) {
            d_digit_offsets[d] = offset;
            offset += s_digit_totals[d];
        }
    }
}
// Step 3: scatter
/**
 * Scatters elements to their sorted positions using warp-parallel ballot.
 *
 * Key improvements over the sequential warp-0-only approach:
 *  - All NUM_WARPS warps work in parallel throughout (no wasted warp capacity).
 *  - Only 3 __syncthreads() total (vs 128 in the sequential design).
 *  - s_local_ranks eliminated: rank+scatter fused into Phase 3, saving 8 KB.
 *  - 17 KB shared memory total → 2 blocks per SM on 48 KB hardware.
 *
 * Warp assignment: warp w owns groups {w, w+NUM_WARPS, w+2*NUM_WARPS, ...}.
 * Each group covers 32 elements in s_digits.  The interleaved stride preserves
 * input order across warps, which is required for stable rank computation.
 *
 * Rank formula for element at pos (warp w, group g, lane l):
 *   rank = s_warp_prefix[w][d]          (all preceding warps' digit-d count)
 *        + intra_count[d]               (warp w's prior groups' digit-d count)
 *        + __popc(ballot & lane_mask)   (digit-d elements before lane l in this group)
 *
 * Shared memory: 2*(SECTION_SIZE*COARSE_FACTOR) + NUM_WARPS*RADIX uints (~17 KB)
 */
__global__ void scatter(unsigned int* d_out,
    unsigned int* d_in,
    unsigned int* d_block_offsets,
    unsigned int* d_digit_offsets,
    unsigned int shift,
    unsigned int n)
{
    constexpr int NUM_WARPS = SECTION_SIZE / 32;

    __shared__ unsigned int s_values[SECTION_SIZE * COARSE_FACTOR];  // 8 KB
    __shared__ unsigned int s_digits[SECTION_SIZE * COARSE_FACTOR];  // 8 KB
    __shared__ unsigned int s_warp_prefix[NUM_WARPS * RADIX];        // 1 KB

    int tid  = threadIdx.x;
    int bid  = blockIdx.x;
    int lane = tid % 32;
    int warp = tid / 32;

    int base_idx   = bid * SECTION_SIZE * COARSE_FACTOR;
    int block_size = min(SECTION_SIZE * COARSE_FACTOR, (int)n - base_idx);
    int total_groups = (block_size + 31) / 32;

    // ----------------------------------------------------------------
    // Load + Init  (Sync 1)
    // ----------------------------------------------------------------
    for (int i = 0; i < COARSE_FACTOR; i++) {
        int local_idx  = tid + i * SECTION_SIZE;
        int global_idx = base_idx + local_idx;
        if (local_idx < block_size) {
            unsigned int val = d_in[global_idx];
            s_values[local_idx] = val;
            s_digits[local_idx] = (val >> shift) & (RADIX - 1);
        }
    }
    if (tid < NUM_WARPS * RADIX)
        s_warp_prefix[tid] = 0;
    __syncthreads();  // Sync 1: load and zero-init both visible

    // ----------------------------------------------------------------
    // Phase 1: Each warp counts its own groups into s_warp_prefix[warp][digit].
    // Blocked assignment: warp w owns groups [w*COARSE_FACTOR, (w+1)*COARSE_FACTOR).
    // This guarantees all of warp w-1's positions precede all of warp w's positions
    // in input order, so the inter-warp prefix sum in Phase 2 is correct.
    // ----------------------------------------------------------------
    int g_start = warp * COARSE_FACTOR;
    int g_end   = min(g_start + COARSE_FACTOR, total_groups);

    for (int g = g_start; g < g_end; g++) {
        int  pos    = g * 32 + lane;
        bool active = (pos < block_size);
        unsigned int digit = active ? s_digits[pos] : 0xFFFFFFFFu;

        for (unsigned int d = 0; d < RADIX; d++) {
            unsigned int ballot = __ballot_sync(0xFFFFFFFFu, active && (digit == d));
            if (lane == 0)
                s_warp_prefix[warp * RADIX + d] += __popc(ballot);
        }
    }
    __syncthreads();  // Sync 2: all warp counts visible for inter-warp scan

    // ----------------------------------------------------------------
    // Phase 2: RADIX threads compute inter-warp exclusive prefix sums.
    // Overwrites s_warp_prefix[w][d] from "count" to "offset".
    // ----------------------------------------------------------------
    if (tid < RADIX) {
        unsigned int running = 0;
        for (int w = 0; w < NUM_WARPS; w++) {
            unsigned int cnt = s_warp_prefix[w * RADIX + tid];
            s_warp_prefix[w * RADIX + tid] = running;
            running += cnt;
        }
    }
    __syncthreads();  // Sync 3: offsets visible to all warps for Phase 3

    // ----------------------------------------------------------------
    // Phase 3 + Scatter: Each warp recomputes ranks and writes to d_out.
    // Fused to avoid storing s_local_ranks (-8 KB shared memory).
    // intra_count[d] lives in lane-0's registers; broadcast via __shfl_sync.
    // Each warp writes only its own positions → no cross-warp hazard.
    // ----------------------------------------------------------------
    unsigned int intra_count[RADIX] = {};  // registers; lane 0 maintains, others broadcast

    for (int g = g_start; g < g_end; g++) {
        int  pos    = g * 32 + lane;
        bool active = (pos < block_size);
        unsigned int digit = active ? s_digits[pos] : 0xFFFFFFFFu;

        for (unsigned int d = 0; d < RADIX; d++) {
            unsigned int ballot = __ballot_sync(0xFFFFFFFFu, active && (digit == d));
            unsigned int rank_in_group = __popc(ballot & ((1u << lane) - 1u));
            // Broadcast lane 0's running count to all lanes in the warp.
            unsigned int base = __shfl_sync(0xFFFFFFFFu, intra_count[d], 0);

            if (active && digit == d) {
                unsigned int out_pos = d_digit_offsets[d]
                                     + d_block_offsets[d * gridDim.x + bid]
                                     + s_warp_prefix[warp * RADIX + d]
                                     + base + rank_in_group;
                d_out[out_pos] = s_values[pos];
            }
            if (lane == 0)
                intra_count[d] += __popc(ballot);
        }
    }
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
extern "C" void radix_sort_iteration(unsigned int* d_out,
    unsigned int* d_in,
    unsigned int* d_hist,
    unsigned int* d_block_offsets,
    unsigned int* d_digit_offsets,
    unsigned int shift,
    unsigned int n)
{
    // Calculate grid size
    // Each block processes SECTION_SIZE * COARSE_FACTOR elements due to thread coarsening
    int items_per_block = SECTION_SIZE * COARSE_FACTOR;
    int num_blocks = (n + items_per_block - 1) / items_per_block;
    num_blocks = min(num_blocks, 65535);  // Safety limit

    // Step 1: Compute histogram
    compute_histogram << <num_blocks, SECTION_SIZE >> > (d_in, d_hist, shift, n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 2: GPU prefix sum
    compute_global_offsets << <1, RADIX >> > (d_hist, d_block_offsets, d_digit_offsets, num_blocks);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 3: Scatter with selected kernel
    scatter << <num_blocks, SECTION_SIZE >> >
        (d_out, d_in, d_block_offsets, d_digit_offsets, shift, n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

}
