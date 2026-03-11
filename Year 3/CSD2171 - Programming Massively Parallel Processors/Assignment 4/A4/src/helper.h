/*
* Copyright 2026 Digipen.  All rights reserved.
*
* Please refer to the end user license associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms
* is strictly prohibited.
*
*/
#pragma once
#ifndef HELPER_H
#define HELPER_H
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

////////////////////////////////////////////////////////////////////////////////
// Common definitions
////////////////////////////////////////////////////////////////////////////////
// Radix sort configuration
#define SECTION_SIZE 512
#define RADIX_BITS 4  // Can be 1, 2, 4, 8
#define RADIX (1 << RADIX_BITS)
#define COARSE_FACTOR 4  // Thread coarsening factor


// Host wrapper for radix sort iteration
extern "C" void radix_sort_iteration_base(unsigned int* d_out,
    unsigned int* d_in,
    unsigned int* d_hist,
    unsigned int* d_block_offsets,
    unsigned int* d_digit_offsets,
    unsigned int shift,
    unsigned int n);

extern "C" void radix_sort_iteration(unsigned int* d_out,
    unsigned int* d_in,
    unsigned int* d_hist,
    unsigned int* d_block_offsets,
    unsigned int* d_digit_offsets,
    unsigned int shift,
    unsigned int n);

extern "C" bool verify(unsigned int* data, int n);

extern "C" float time_radix_sort(unsigned int* d_data, unsigned int n, int kernel_version);

#endif