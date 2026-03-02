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

}

// CUDA Kernel : Scan Phase 1: Block-level exclusive scan with proper boundary handling
// /! \brief Performs a block-level exclusive scan (Phase 1) for large-scale arrays. 
// \param data Device pointer to the data array to be scanned in-place. 
// \param block_sums Device pointer to store the total sum of each block for global synchronization. 
// \param n Total number of elements to scan in the global array. /
__global__ void blockExclusiveScanPhase1(int* data, int* block_sums, int n)
{

}

// CUDA Kernel : Scan Phase 2: Scan the block sums
// /! \brief Scans the intermediate block sums (Phase 2) to compute global offsets. 
// \param block_sums Device pointer to the array of block sums to be scanned. 
// \param num_blocks Number of blocks generated in Phase 1. /
__global__ void scanBlockSums(int* block_sums, int num_blocks)
{

}

// CUDA Kernel : Scan Phase 3: Add scanned block sums to each block
// /! \brief Adds scanned block offsets to each local block result (Phase 3). 
// \param data Device pointer to the local scanned data to be updated. 
// \param block_sums Device pointer to the scanned global block offsets. 
// \param n Total number of elements in the data array. /
__global__ void addBlockSumsPhase3(int* data, const int* block_sums, int n)
{

}

// Wrapper function for multi-block exclusive scan
// /! \brief Orchestrates a robust multi-block exclusive prefix sum on the GPU. 
// \param d_data Device pointer to the global array to be scanned in-place. 
// \param n Number of elements in the array. /
void exclusiveScanMultiBlock(int* d_data, int n)
{

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

