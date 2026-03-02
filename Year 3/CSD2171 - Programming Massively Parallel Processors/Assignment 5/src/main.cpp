#include <iostream>
#include "helper.h"

// Main test program
int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Only one arugment is required" << std::endl;
        return 1;
    }

    int choice = atoi(argv[1]);

    switch (choice)
    {
    case 0:
        // Test the scan function first
        if (!ScanTester::testScan()) {
            std::cerr << "Scan test failed! Aborting." << std::endl;
            return 1;
        }
        break;

    case 1:
        // Test 1: Fixed example matrix
        std::cout << "\n=== Test 1: Fixed Example Matrix ===" << std::endl;
        {
            SparseMatrixCOO coo_matrix(4, 4);
            coo_matrix.add_element(0, 0, 1.0f);
            coo_matrix.add_element(0, 2, 7.0f);
            coo_matrix.add_element(1, 2, 8.0f);
            coo_matrix.add_element(2, 1, 4.0f);
            coo_matrix.add_element(2, 2, 3.0f);
            coo_matrix.add_element(3, 0, 2.0f);
            coo_matrix.add_element(3, 3, 1.0f);

            COOtoCSRConverter converter;
            SparseMatrixCSR csr_matrix = converter.convert(coo_matrix);
            csr_matrix.print();

            ConversionValidator::validate(coo_matrix, csr_matrix);
        }
        break;

    case 2:
        // Test 2: Random sparse matrix
        std::cout << "\n=== Test 2: Random Sparse Matrix ===" << std::endl;
        {
            RandomSparseMatrixGenerator generator(42); // Fixed seed for reproducibility

            // Test with moderate size
            int num_rows = 1000;
            int num_cols = 1000;
            float density = 0.01f; // 1% density

            SparseMatrixCOO random_coo = generator.generate(num_rows, num_cols, density);

            COOtoCSRConverter converter(256, true);
            cudaEvent_t start, stop;
            float ms = 0.0f;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            SparseMatrixCSR random_csr = converter.convert(random_coo);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms, start, stop);
            std::cout << "CUDA conversion completed in " << ms << " ms" << std::endl;
            random_csr.print(15);

            ConversionValidator::validate(random_coo, random_csr);
        }
        break;

    case 3:
        // Test 3: Random sparse matrix
        std::cout << "\n=== Test 3: Larger Random Sparse Matrix ===" << std::endl;
        {
            RandomSparseMatrixGenerator generator(42); // Fixed seed for reproducibility

            // Test with larger size
            int num_rows = 10000;
            int num_cols = 10000;
            float density = 0.01f; // 1% density

            SparseMatrixCOO random_coo = generator.generate(num_rows, num_cols, density);

            COOtoCSRConverter converter(256, true);
            cudaEvent_t start, stop;
            float ms = 0.0f;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            SparseMatrixCSR random_csr = converter.convert(random_coo);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms, start, stop);
            std::cout << "CUDA conversion completed in " << ms << " ms" << std::endl;
            random_csr.print(15);

            ConversionValidator::validate(random_coo, random_csr);
        }
        break;

    case 4:
        // Test 4: Larger random sparse matrix
        std::cout << "\n=== Test 4: Largest Random Sparse Matrix ===" << std::endl;
        {
            RandomSparseMatrixGenerator generator(123);

            int num_rows = 10000;
            int num_cols = 10000;
            float density = 0.05f; // density

            SparseMatrixCOO large_coo = generator.generate(num_rows, num_cols, density);

            COOtoCSRConverter converter(256, true);
            cudaEvent_t start, stop;
            float ms = 0.0f;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            SparseMatrixCSR large_csr = converter.convert(large_coo);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms, start, stop);
            std::cout << "CUDA conversion completed in " << ms << " ms" << std::endl;
            large_csr.print(10);
            
			// You can enable the following for element-wise checking
            // ConversionValidator::validate(large_coo, large_csr);
			
            // For large matrices, do a quick validation
            std::cout << "\nQuick validation for large matrix..." << std::endl;
            int total_nnz = large_csr.row_ptr[num_rows];
            if (total_nnz == large_coo.nnz) {
                std::cout << "Total nnz matches: " << total_nnz << std::endl;
            }
            else {
                std::cout << "Error: Total nnz mismatch: COO has " << large_coo.nnz
                    << ", CSR has " << total_nnz << std::endl;
            }
        }
        break;
    }

    //std::cout << "\nAll tests completed successfully!" << std::endl;

    // Reset CUDA device
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}