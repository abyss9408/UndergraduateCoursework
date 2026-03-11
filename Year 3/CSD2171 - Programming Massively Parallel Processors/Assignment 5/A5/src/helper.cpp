#include <iostream>
#include "helper.h"
using namespace std;

// Sparse matrix representation in COO format
void SparseMatrixCOO::add_element(int row, int col, float val) {
    rows.push_back(row);
    cols.push_back(col);
    values.push_back(val);
    nnz++;
}

void SparseMatrixCOO::clear() {
    rows.clear();
    cols.clear();
    values.clear();
    nnz = 0;
}


// Sparse matrix representation in CSR format

void SparseMatrixCSR::clear() {
    row_ptr.clear();
    col_ind.clear();
    values.clear();
    row_ptr.resize(num_rows + 1, 0);
    nnz = 0;
}

void SparseMatrixCSR::print(int max_elements ) const {
    std::cout << "\nCSR Format (showing first " << max_elements << " entries):" << std::endl;

    std::cout << "row_ptr: ";
    int print_rows = std::min(num_rows + 1, max_elements);
    for (int i = 0; i < print_rows; i++) {
        std::cout << row_ptr[i] << " ";
    }
    if (num_rows + 1 > max_elements) std::cout << "...";

    std::cout << "\ncol_ind: ";
    int print_nnz = std::min(nnz, max_elements);
    for (int i = 0; i < print_nnz; i++) {
        std::cout << col_ind[i] << " ";
    }
    if (nnz > max_elements) std::cout << "...";

    std::cout << "\nvalues: ";
    for (int i = 0; i < print_nnz; i++) {
        std::printf("%.2f ", values[i]);
    }
    if (nnz > max_elements) std::cout << "...";
    std::cout << std::endl;

    std::cout << "Total non-zeros: " << nnz << std::endl;
}

// Random sparse matrix generator
SparseMatrixCOO RandomSparseMatrixGenerator::generate(int num_rows, int num_cols, float density) {
    SparseMatrixCOO matrix(num_rows, num_cols);

    // Calculate expected number of non-zeros
    int expected_nnz = static_cast<int>(num_rows * num_cols * density);
    expected_nnz = std::max(1, std::min(expected_nnz, num_rows * num_cols));

    std::uniform_int_distribution<int> row_dist(0, num_rows - 1);
    std::uniform_int_distribution<int> col_dist(0, num_cols - 1);

    // Generate random non-zero elements
    for (int i = 0; i < expected_nnz; i++) {
        int row = row_dist(rng);
        int col = col_dist(rng);
        float val = dist(rng);
        matrix.add_element(row, col, val);
    }

    std::cout << "Generated random sparse matrix:" << std::endl;
    std::cout << "  Dimensions: " << num_rows << " x " << num_cols << std::endl;
    std::cout << "  Density: " << (density * 100) << "%" << std::endl;
    std::cout << "  Number of non-zeros: " << matrix.nnz << std::endl;

    return matrix;
}


// Scan function tester
bool ScanTester::testScan(int test_size) {
    std::cout << "\n=== Testing Scan Function ===" << std::endl;

    std::vector<int> h_data(test_size);
    std::vector<int> h_expected(test_size);
    std::vector<int> h_result(test_size);

    // Generate test data
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 9);
    for (int i = 0; i < test_size; i++) {
        h_data[i] = dist(rng);
    }

    // Compute expected result on host
    h_expected[0] = 0;
    for (int i = 1; i < test_size; i++) {
        h_expected[i] = h_expected[i - 1] + h_data[i - 1];
    }

    // Allocate device memory
    int* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, test_size * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), test_size * sizeof(int), cudaMemcpyHostToDevice));

    // Test multi-block scan
    exclusiveScanMultiBlock(d_data, test_size);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_data, test_size * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify
    bool success = verifyScanResult(h_data, h_result, test_size);
    std::cout << "Multi-block scan: " << (success ? "PASS" : "FAIL") << std::endl;

    CUDA_CHECK(cudaFree(d_data));
    std::cout << "Scan test completed." << std::endl;

    return success;
}


bool ScanTester::verifyScanResult(const std::vector<int>& data, const std::vector<int>& result, int n) {
    int running_sum = 0;
    for (int i = 0; i < n; i++) {
        if (result[i] != running_sum) {
            std::cout << "Scan verification failed at index " << i
                << ": expected " << running_sum << ", got " << result[i] << std::endl;
            return false;
        }
        running_sum += data[i];
    }
    return true;
}

// Conversion validator

bool ConversionValidator::validate(const SparseMatrixCOO& coo, const SparseMatrixCSR& csr) {
    std::cout << "\nValidating conversion..." << std::endl;

    // Count non-zeros per row in COO
    std::vector<int> row_counts(csr.num_rows, 0);
    for (int i = 0; i < coo.nnz; i++) {
        if (coo.rows[i] >= 0 && coo.rows[i] < csr.num_rows) {
            row_counts[coo.rows[i]]++;
        }
    }

    bool valid = true;
    int total_errors = 0;
    int total_nnz = csr.row_ptr[csr.num_rows];

    // Check total nnz
    int expected_total_nnz = 0;
    for (int count : row_counts) {
        expected_total_nnz += count;
    }

    if (total_nnz != expected_total_nnz) {
        std::cout << "Error: Total nnz mismatch: expected " << expected_total_nnz
            << ", got " << total_nnz << std::endl;
        valid = false;
        total_errors++;
    }

    // Check each row
    for (int i = 0; i < csr.num_rows; i++) {
        int expected = row_counts[i];
        int actual = csr.row_ptr[i + 1] - csr.row_ptr[i];

        if (expected != actual) {
            if (total_errors < 5) {
                std::cout << "Error at row " << i << ": expected " << expected
                    << ", got " << actual << std::endl;
            }
            valid = false;
            total_errors++;
        }
    }

    if (total_errors >= 5) {
        std::cout << "... and " << (total_errors - 5) << " more errors" << std::endl;
    }

    if (valid) {
        std::cout << "Conversion validated successfully!" << std::endl;
    }
    else {
        std::cout << "Conversion validation failed with " << total_errors << " errors!" << std::endl;
    }

    return valid;
}

extern "C" void exclusiveScanMultiBlock(int* d_data, int n);




