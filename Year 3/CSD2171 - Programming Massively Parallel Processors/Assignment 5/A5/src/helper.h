#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
// Helper function for CUDA error checking
#define CUDA_CHECK(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
                  << ": " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}
// // Sparse matrix representation in COO format
struct SparseMatrixCOO {
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<float> values;
    int num_rows;
    int num_cols;
    int nnz;

    SparseMatrixCOO(int m, int n) : num_rows(m), num_cols(n), nnz(0) {}

    void add_element(int row, int col, float val);

    void clear();
};

// Sparse matrix representation in CSR format
struct SparseMatrixCSR {
    std::vector<int> row_ptr;
    std::vector<int> col_ind;
    std::vector<float> values;
    int num_rows;
    int num_cols;
    int nnz;

    SparseMatrixCSR(int m, int n) : num_rows(m), num_cols(n), nnz(0) {
        row_ptr.resize(m + 1, 0);
    }

    void clear();

    void print(int max_elements = 20) const;
};

// Random sparse matrix generator
class RandomSparseMatrixGenerator {
private:
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist;

public:
    RandomSparseMatrixGenerator(unsigned int seed = std::random_device{}())
        : rng(seed), dist(0.0f, 10.0f) {
    }
    // /! \brief Generates a random sparse matrix in COO format. 
    // \param num_rows Number of rows in the matrix. 
    // \param num_cols Number of columns in the matrix. 
    // \param density Probability (0.0 to 1.0) of a non-zero element at any position. 
    // \return A SparseMatrixCOO object containing the generated data. 
    // note Uses Move Semantics to efficiently return the matrix by value.
    SparseMatrixCOO generate(int num_rows, int num_cols, float density);
};

// Scan function tester
class ScanTester {
public:
    static bool testScan(int test_size = 10000);

private:
    static bool verifyScanResult(const std::vector<int>& data, const std::vector<int>& result, int n);
};

// Conversion validator
class ConversionValidator {
public:
    static bool validate(const SparseMatrixCOO& coo, const SparseMatrixCSR& csr);
};

// COO to CSR converter using CUDA
class COOtoCSRConverter {
private:
    int block_size;
    bool validate_scan;

public:
    COOtoCSRConverter(int block_size = 256, bool validate_scan = true)
        : block_size(block_size), validate_scan(validate_scan) {
    }

    // Convert COO matrix to CSR format using CUDA
    SparseMatrixCSR convert(const SparseMatrixCOO& coo_matrix);

private:
    // Validate the prefix scan result
    void validatePrefixScan(int* d_csr_row_ptr, 
                            int num_rows, 
                            const std::vector<int>& row_counts);
};



extern "C" void exclusiveScanMultiBlock(int* d_data, int n);
