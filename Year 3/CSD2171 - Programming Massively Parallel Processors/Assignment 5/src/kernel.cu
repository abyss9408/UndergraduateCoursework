/* Start Header *****************************************************************/
/*!
    \file kernel.cu

    \author Bryan Ang Wei Ze, bryanweize.ang, 2301397

    \par bryanweize.ang\@digipen.edu

    \date March 5, 2026

    \brief Copyright (C) 2026 DigiPen Institute of Technology.

    Reproduction or disclosure of this file or its contents without the prior written consent of DigiPen Institute of Technology is prohibited.
*/
/* End Header *******************************************************************/
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "helper.h"

// CUDA Kernel : This kernel computes the sum of all elements in the input array 
// using an optimized reduction pattern. It employs shared memory for intra - block 
// reductions and atomic operations for inter - block result aggregation.
// \param in Device pointer to the input array of integers to be summed.
//           Must be allocated with size at least n * sizeof(int) bytes.
//           The array contents remain unchanged by the kernel.
// \param out Device pointer to the output array that stores reduction results.
//           Must be allocated with size at least gridDim.x * sizeof(int) bytes.
// Output format:
// - out[0]: Final sum of all elements (only valid after host synchronization)
// - out[blockIdx.x]: Partial sum from block with index blockIdx.x [optioanl]
// For single-block launches, out[0] contains the complete sum.
// \param n Number of elements in the input array.
//         Must be a positive integer. For best performance, n should be
//         a multiple of the block size and warp size (32)./

__global__ void reduceSumKernel(const int* in, int* out, const int n)
{
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Load: pad out-of-bounds threads with 0
    sdata[tid] = (gid < n) ? in[gid] : 0;
    __syncthreads();

    // Reduction: stride halves each step
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    // Thread 0 of each block writes its partial sum
    // atomicAdd accumulates all blocks into out[0]
    if (tid == 0)
        atomicAdd(&out[0], sdata[0]);
}

// CUDA Kernel : Count non-zeros per row
// /! \brief Counts the number of non-zero elements per row from COO format. 
// \param coo_rows Device pointer to the input array of row indices. 
// \param row_counts Device pointer to the output array for counting non-zeros per row. 
// \param nnz Total number of non-zero elements in the matrix. 
// \param num_rows Total number of rows in the matrix. /
__global__ void countNonzerosPerRow(const int* coo_rows,
    int* row_counts,
    int nnz,
    int num_rows)
{
    // Calculate global thread ID
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int sride = blockDim.x * gridDim.x;

    // Iterate through all NNZ, processing in strides for coalesced access
    for (int i = thread_id; i < nnz; i += sride)
    {
        int row = coo_rows[i];

        if (row >= 0 && row < num_rows)
        {
            // Atomically increment the count for the corresponding row
            atomicAdd(&row_counts[row], 1);
        }
    }
}

// CUDA Kernel : Scan Phase 1: Block-level exclusive scan with proper boundary handling
// /! \brief Performs a block-level exclusive scan (Phase 1) for large-scale arrays. 
// \param data Device pointer to the data array to be scanned in-place. 
// \param block_sums Device pointer to store the total sum of each block for global synchronization. 
// \param n Total number of elements to scan in the global array. /
__global__ void blockExclusiveScanPhase1(int* data, int* block_sums, int n)
{
    extern __shared__ int s[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Load from global memory into shared memory
    // If this thread's global index is within bounds, load the real value.
    // Otherwise load 0 (padding so the math still works on non-power-of-2 sizes).
    s[tid] = (gid < n) ? data[gid] : 0;
    __syncthreads();

    // Up-sweep (reduction phase)
    // stride doubles each iteration: 1, 2, 4, 8, ...
    // At each stride, thread at position (stride*2 - 1) accumulates from (stride - 1).
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        // Which threads are active this round?
        // Only threads whose index satisfies: (tid+1) is a multiple of (2*stride)
        int index = (tid + 1) * 2 * stride - 1;
        if (index < blockDim.x)
        {
            s[index] += s[index - stride];
        }
        __syncthreads();
    }

    // Save block sum BEFORE zeroing
    // The last element of shared memory now holds the TOTAL sum of this block.
    if (tid == 0)
    {
        block_sums[blockIdx.x] = s[blockDim.x - 1];
        s[blockDim.x - 1] = 0;   // set to 0 to start the down-sweep
    }
    __syncthreads();

    // Down-sweep phase
    // stride halves each iteration: blockDim.x/2, blockDim.x/4, ..., 1
    for (int stride = blockDim.x / 2; stride >= 1; stride /= 2)
    {
        int index = (tid + 1) * 2 * stride - 1;
        if (index < blockDim.x)
        {
            int left = index - stride;
            int temp = s[left];
            s[left] = s[index];        // left child gets parent's value
            s[index] += temp;          // right child gets parent + left child
        }
        __syncthreads();
    }

    // Write result back to global memory
    if (gid < n)
    {
        data[gid] = s[tid];
    }
}

// CUDA Kernel : Scan Phase 2: Scan the block sums
// /! \brief Scans the intermediate block sums (Phase 2) to compute global offsets. 
// \param block_sums Device pointer to the array of block sums to be scanned. 
// \param num_blocks Number of blocks generated in Phase 1. /
__global__ void scanBlockSums(int* block_sums, int num_blocks)
{
    extern __shared__ int s[];
    int tid = threadIdx.x;

    // Load: each thread loads one block_sum value (or 0 if out of range)
    s[tid] = (tid < num_blocks) ? block_sums[tid] : 0;
    __syncthreads();

    // Up-sweep
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        int index = (tid + 1) * 2 * stride - 1;
        if (index < blockDim.x)
        {
            s[index] += s[index - stride];
        }
        __syncthreads();
    }

    // Zero the last element → begin down-sweep
    if (tid == 0) s[blockDim.x - 1] = 0;
    __syncthreads();

    // Down-sweep
    for (int stride = blockDim.x / 2; stride >= 1; stride /= 2)
    {
        int index = (tid + 1) * 2 * stride - 1;
        if (index < blockDim.x)
        {
            int left = index - stride;
            int temp = s[left];
            s[left] = s[index];
            s[index] += temp;
        }
        __syncthreads();
    }

    // Write back
    if (tid < num_blocks)
    {
        block_sums[tid] = s[tid];
    }
}

// CUDA Kernel : Scan Phase 3: Add scanned block sums to each block
// /! \brief Adds scanned block offsets to each local block result (Phase 3). 
// \param data Device pointer to the local scanned data to be updated. 
// \param block_sums Device pointer to the scanned global block offsets. 
// \param n Total number of elements in the data array. /
__global__ void addBlockSumsPhase3(int* data, const int* block_sums, int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Block 0 has offset 0 — adding 0 changes nothing, so we can skip it
    // but the code below handles it correctly either way.
    if (gid < n && blockIdx.x > 0)
    {
        data[gid] += block_sums[blockIdx.x];
    }
}

// Wrapper function for multi-block exclusive scan
// /! \brief Orchestrates a robust multi-block exclusive prefix sum on the GPU. 
// \param d_data Device pointer to the global array to be scanned in-place. 
// \param n Number of elements in the array. /
void exclusiveScanMultiBlock(int* d_data, int n)
{
    const int BLOCK_SIZE = 256;
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Phase 1
    // Each block scans its own chunk; block totals go into d_block_sums.
    int* d_block_sums;
    cudaMalloc(&d_block_sums, num_blocks * sizeof(int));
    cudaMemset(d_block_sums, 0, num_blocks * sizeof(int));

    size_t shared_bytes = BLOCK_SIZE * sizeof(int);
    blockExclusiveScanPhase1 << <num_blocks, BLOCK_SIZE, shared_bytes >> > (
        d_data, d_block_sums, n
        );
    cudaDeviceSynchronize();

    // Phase 2
    // Scan the block_sums array.
    // If num_blocks > BLOCK_SIZE we would need to recurse — for typical matrices
    // (up to ~2.5 million rows with BLOCK_SIZE=256) num_blocks stays well under 256.
    // Round up to next power of 2 so the Blelloch kernel works correctly.
    int scan_size = 1;
    while (scan_size < num_blocks) scan_size *= 2;

    scanBlockSums << <1, scan_size, scan_size * sizeof(int) >> > (
        d_block_sums, num_blocks
        );
    cudaDeviceSynchronize();

    // Phase 3
    // Add each block's global offset to its local results.
    addBlockSumsPhase3 << <num_blocks, BLOCK_SIZE >> > (
        d_data, d_block_sums, n
        );
    cudaDeviceSynchronize();

    cudaFree(d_block_sums);
}

// CUDA Kernel : Scatter COO data into CSR format
// /! \brief Scatters COO data elements into their respective CSR positions. 
// \param coo_rows Device pointer to input COO row indices. 
// \param coo_cols Device pointer to input COO column indices. 
// \param coo_vals Device pointer to input COO values. 
// \param csr_row_ptr Device pointer to scanned row pointers used for positioning. 
// \param csr_col_ind Device pointer to output CSR column index array. 
// \param csr_vals Device pointer to output CSR value array. 
// \param nnz Total number of non-zero elements. 
// \param num_rows Total number of rows in the matrix. /
__global__ void scatterCOOtoCSR(const int* coo_rows,
    const int* coo_cols,
    const float* coo_vals,
    int* csr_row_ptr,
    int* csr_col_ind,
    float* csr_vals,
    int nnz,
    int num_rows)
{
    // Each thread handles one COO element
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Stride loop: handles cases where nnz > total number of threads launched
    for (; i < nnz; i += blockDim.x * gridDim.x)
    {
        int row = coo_rows[i];

        // Bounds check: skip if row index is invalid
        if (row < 0 || row >= num_rows) continue;

        // atomicAdd returns the OLD value (current slot), then increments by 1.
        // This gives this thread a unique destination slot for its element.
        int pos = atomicAdd(&csr_row_ptr[row], 1);

        // Write the COO element into its CSR destination
        csr_col_ind[pos] = coo_cols[i];
        csr_vals[pos] = coo_vals[i];
    }
}

// CUDA Kernel : Fix row pointers after scattering
// /! \brief Adjusts row pointers to their original state after they were modified by atomic operations. 
// \param csr_row_ptr Device pointer to the row pointers that need correction. 
// \param row_counts Device pointer to the original non-zero counts per row. 
// \param num_rows Total number of rows in the matrix. /
__global__ void fixRowPointers(int* csr_row_ptr,
    const int* row_counts,
    int num_rows)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows)
    {
        // After scatter: ptr[row] = original_start + row_counts[row]
        // We want:       ptr[row] = original_start
        // So:            ptr[row] -= row_counts[row]
        csr_row_ptr[row] -= row_counts[row];
    }
}

// CUDA Kernel : Sort columns within each row (odd-even sort / bitonic sort for GPU efficiency)
// /! \brief Sorts column indices within each CSR row using odd-even sort or a bitonic sorting network. 
// \param csr_row_ptr Device pointer to the row pointers defining row boundaries. 
// \param csr_col_ind Device pointer to CSR column indices to be sorted in-place. 
// \param csr_vals Device pointer to CSR values to be swapped alongside indices. 
// \param row_counts Device pointer to the number of elements in each row. 
// \param num_rows Total number of rows in the matrix. /
__global__ void sortRows(const int* csr_row_ptr,
    int* csr_col_ind,
    float* csr_vals,
    const int* row_counts,
    int num_rows)
{
    // One THREAD handles one row (vs. old design: one BLOCK per row)
    // This lets us use the standard row_grid_size launch with no shared memory.
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    int start = csr_row_ptr[row];
    int count = row_counts[row];

    if (count <= 1) return;  // 0 or 1 elements: already sorted

    // Insertion sort on this row's slice of global memory.
    // Sparse rows are short (typically single-digit element counts),
    // so O(count^2) serial work per thread is negligible in practice.
    for (int i = 1; i < count; i++)
    {
        int   key_col = csr_col_ind[start + i];
        float key_val = csr_vals[start + i];

        int j = i - 1;
        while (j >= 0 && csr_col_ind[start + j] > key_col)
        {
            csr_col_ind[start + j + 1] = csr_col_ind[start + j];
            csr_vals[start + j + 1] = csr_vals[start + j];
            j--;
        }
        csr_col_ind[start + j + 1] = key_col;
        csr_vals[start + j + 1] = key_val;
    }
}


// Convert COO matrix to CSR format using CUDA
// /! \brief High-level API to convert a COO matrix into CSR format using CUDA. 
// \param coo_matrix The input SparseMatrixCOO structure on the host. 
// \return A SparseMatrixCSR structure containing the converted matrix data. 
// \note Handles all device memory management and kernel dispatching. /
SparseMatrixCSR  COOtoCSRConverter::convert(const SparseMatrixCOO& coo_matrix) {
    int nnz = coo_matrix.nnz;
    int num_rows = coo_matrix.num_rows;
    int num_cols = coo_matrix.num_cols;

    // Allocate device memory for COO data
    int* d_coo_rows, * d_coo_cols;
    float* d_coo_vals;
    CUDA_CHECK(cudaMalloc(&d_coo_rows, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_coo_cols, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_coo_vals, nnz * sizeof(float)));

    // Copy COO data to device
    CUDA_CHECK(cudaMemcpy(d_coo_rows, coo_matrix.rows.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_coo_cols, coo_matrix.cols.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_coo_vals, coo_matrix.values.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));

    // Allocate device memory for intermediate arrays
    int* d_row_counts, * d_csr_row_ptr, * d_csr_col_ind;
    float* d_csr_vals;
    CUDA_CHECK(cudaMalloc(&d_row_counts, num_rows * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csr_row_ptr, (num_rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csr_col_ind, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csr_vals, nnz * sizeof(float)));

    // Initialize arrays
    CUDA_CHECK(cudaMemset(d_row_counts, 0, num_rows * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_csr_row_ptr, 0, (num_rows + 1) * sizeof(int)));

    // Step 1: Count non-zeros per row
    int grid_size = (nnz + block_size - 1) / block_size;
    countNonzerosPerRow << <grid_size, block_size >> > (d_coo_rows, d_row_counts, nnz, num_rows);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 2: Compute row pointers using exclusive scan
    CUDA_CHECK(cudaMemcpy(d_csr_row_ptr, d_row_counts, num_rows * sizeof(int), cudaMemcpyDeviceToDevice));

    //std::cout << "Performing multi-block scan on " << num_rows << " elements..." << std::endl;
    exclusiveScanMultiBlock(d_csr_row_ptr, num_rows);

    // Set last element to total nnz
    int* d_total_nnz = &d_csr_row_ptr[num_rows];
    int num_blocks = (num_rows + 256 - 1) / 256;
    int shared_mem_size = 256 * sizeof(int);
    int h_total_nnz;
    // reduction
    reduceSumKernel << <num_blocks, 256, shared_mem_size >> > (d_row_counts, d_total_nnz, num_rows);
    CUDA_CHECK(cudaMemcpy(&h_total_nnz, d_total_nnz, sizeof(int), cudaMemcpyDeviceToHost));

    int total_nnz = h_total_nnz;
    if (validate_scan) {
        std::vector<int> h_row_counts(num_rows);
        CUDA_CHECK(cudaMemcpy(h_row_counts.data(), d_row_counts, num_rows * sizeof(int), cudaMemcpyDeviceToHost));
        validatePrefixScan(d_csr_row_ptr, num_rows, h_row_counts);
    }

    // Step 3: Create a copy of csr_row_ptr for scattering
    int* d_csr_row_ptr_copy;
    CUDA_CHECK(cudaMalloc(&d_csr_row_ptr_copy, (num_rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_csr_row_ptr_copy, d_csr_row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyDeviceToDevice));

    // Step 4: Scatter COO data into CSR format
    scatterCOOtoCSR << <grid_size, block_size >> > (
        d_coo_rows, d_coo_cols, d_coo_vals,
        d_csr_row_ptr_copy, d_csr_col_ind, d_csr_vals,
        nnz, num_rows
        );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 5: Fix row pointers
    int row_grid_size = (num_rows + block_size - 1) / block_size;
    fixRowPointers << <row_grid_size, block_size >> > (d_csr_row_ptr_copy, d_row_counts, num_rows);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 6: Sort columns within each row
    sortRows << <row_grid_size, block_size >> > (
        d_csr_row_ptr_copy, d_csr_col_ind, d_csr_vals,
        d_row_counts, num_rows
        );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Create result CSR matrix
    SparseMatrixCSR csr_matrix(num_rows, num_cols);
    csr_matrix.nnz = total_nnz;

    // Copy results back to host
    csr_matrix.row_ptr.resize(num_rows + 1);
    csr_matrix.col_ind.resize(total_nnz);
    csr_matrix.values.resize(total_nnz);

    CUDA_CHECK(cudaMemcpy(csr_matrix.row_ptr.data(), d_csr_row_ptr_copy, (num_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(csr_matrix.col_ind.data(), d_csr_col_ind, total_nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(csr_matrix.values.data(), d_csr_vals, total_nnz * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_coo_rows));
    CUDA_CHECK(cudaFree(d_coo_cols));
    CUDA_CHECK(cudaFree(d_coo_vals));
    CUDA_CHECK(cudaFree(d_row_counts));
    CUDA_CHECK(cudaFree(d_csr_row_ptr));
    CUDA_CHECK(cudaFree(d_csr_row_ptr_copy));
    CUDA_CHECK(cudaFree(d_csr_col_ind));
    CUDA_CHECK(cudaFree(d_csr_vals));

    return csr_matrix;
}

void COOtoCSRConverter::validatePrefixScan(int* d_csr_row_ptr, int num_rows, const std::vector<int>& row_counts) {
    std::vector<int> h_row_ptr(num_rows + 1);
    CUDA_CHECK(cudaMemcpy(h_row_ptr.data(), d_csr_row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost));

    bool scan_valid = true;
    for (int i = 0; i < num_rows; i++) {
        if (h_row_ptr[i + 1] < h_row_ptr[i]) {
            std::cout << "Warning: Scan invalid at index " << i
                << ": " << h_row_ptr[i] << " > " << h_row_ptr[i + 1] << std::endl;
            scan_valid = false;
        }
    }

    if (!scan_valid) {
        std::cout << "Scan validation failed! Using fallback host scan..." << std::endl;

        // Compute prefix sum on host
        h_row_ptr[0] = 0;
        for (int i = 0; i < num_rows; i++) {
            h_row_ptr[i + 1] = h_row_ptr[i] + row_counts[i];
        }

        // Copy back to device
        CUDA_CHECK(cudaMemcpy(d_csr_row_ptr, h_row_ptr.data(), (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    }
    else {
        std::cout << "Scan validation passed!" << std::endl;
    }
}

