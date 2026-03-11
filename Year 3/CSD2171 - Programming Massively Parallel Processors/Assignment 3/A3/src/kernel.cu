/* Start Header *****************************************************************/
/*!
    \file kernel.cu

    \author Bryan Ang Wei Ze, bryanweize.ang, 2301397

    \par bryanweize.ang\@digipen.edu

    \date February 18, 2026

    \brief Copyright (C) 2026 DigiPen Institute of Technology.

    Reproduction or disclosure of this file or its contents without the prior written consent of DigiPen Institute of Technology is prohibited.
*/
/* End Header *******************************************************************/
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "histogram_common.h"
#define BLOCK_SIZE 32
#define HISTOGRAM_BINS 256

__global__ void histogram(
    unsigned char* __restrict__ input,
    unsigned int* histo_output,
    float* yuv_output,
    int width,
    int height
)
{
    // Private histogram in shared memory to reduce contention on global memory
    __shared__ unsigned int histo_private[HISTOGRAM_BINS];

    // Linear thread index within the block
    int tx = threadIdx.y * blockDim.x + threadIdx.x;
    int numThreadsPerBlock = blockDim.x * blockDim.y;

    // Initialize private histogram bins to zero (threads cooperate)
    for (int i = tx; i < HISTOGRAM_BINS; i += numThreadsPerBlock)
    {
        histo_private[i] = 0;
    }

    // Synchronize to ensure all bins are initialized before accumulating
    __syncthreads();

    int imgSz = width * height;
    int totalThreads = gridDim.x * numThreadsPerBlock;
    int globalIdx = blockIdx.x * numThreadsPerBlock + tx;

    // Each thread processes pixels with stride, performing RGB->YUV and histogramming
    for (int i = globalIdx; i < imgSz; i += totalThreads)
    {
        // Read RGB values (non-interleaved layout: R plane, G plane, B plane)
        float r = (float)input[i];
        float g = (float)input[i + imgSz];
        float b = (float)input[i + 2 * imgSz];

        // Convert RGB to YUV using BT.601 standard (Eq. 2 in assignment)
        float y = 0.299f * r + 0.587f * g + 0.114f * b;
        float u = -0.169f * r - 0.331f * g + 0.499f * b + 128.0f;
        float v = 0.499f * r - 0.418f * g - 0.0813f * b + 128.0f;

        // Clamp YUV values to [0, 255] and store in output
        y = fminf(fmaxf(y, 0.0f), 255.0f);
        u = fminf(fmaxf(u, 0.0f), 255.0f);
        v = fminf(fmaxf(v, 0.0f), 255.0f);

        yuv_output[i] = y;
        yuv_output[i + imgSz] = u;
        yuv_output[i + 2 * imgSz] = v;

        // Compute histogram bin for the Y component (round to nearest integer)
        int bin = (int)(y + 0.5f);
        bin = min(max(bin, 0), HISTOGRAM_BINS - 1);

        // atomicAdd into local (shared memory) histogram to reduce global contention
        atomicAdd(&histo_private[bin], 1);
    }

    // Wait for all threads in this block to finish their pixel processing
    __syncthreads();

    // Accumulate private histogram into global histogram using atomicAdd
    for (int i = tx; i < HISTOGRAM_BINS; i += numThreadsPerBlock)
    {
        atomicAdd(&histo_output[i], histo_private[i]);
    }
}

__global__ void computeCDFAndMin(
    unsigned int* __restrict__ histo,
    float* cdf_output,
    int width,
    int height
)
{
    // Only block 0 performs the scan (256 bins fit in one block)
    if (blockIdx.x != 0) return;

    int tx = threadIdx.y * blockDim.x + threadIdx.x;

    __shared__ float scan[HISTOGRAM_BINS];
    __shared__ float original[HISTOGRAM_BINS];

    float totalPixels = (float)(width * height);

    // Load histogram bin counts as probabilities (pdf) into shared memory
    if (tx < HISTOGRAM_BINS)
    {
        float val = (float)histo[tx] / totalPixels;
        scan[tx] = val;
        original[tx] = val;
    }
    __syncthreads();

    // ===== Phase 1: Up-sweep (Reduction) =====
    // Build partial sums in a tree structure from leaves to root
    for (int stride = 1; stride < HISTOGRAM_BINS; stride *= 2)
    {
        int index = (tx + 1) * stride * 2 - 1;
        if (index < HISTOGRAM_BINS)
        {
            scan[index] += scan[index - stride];
        }
        __syncthreads();
    }

    // Clear the last element to prepare for exclusive scan
    if (tx == 0)
    {
        scan[HISTOGRAM_BINS - 1] = 0.0f;
    }
    __syncthreads();

    // ===== Phase 2: Down-sweep (Post-reduction) =====
    // Distribute partial sums back down the tree to produce prefix sums
    for (int stride = HISTOGRAM_BINS / 2; stride >= 1; stride /= 2)
    {
        int index = (tx + 1) * stride * 2 - 1;
        if (index < HISTOGRAM_BINS)
        {
            float temp = scan[index - stride];
            scan[index - stride] = scan[index];
            scan[index] += temp;
        }
        __syncthreads();
    }

    // Convert exclusive scan to inclusive: inclusive[i] = exclusive[i] + original[i]
    if (tx < HISTOGRAM_BINS)
    {
        cdf_output[tx] = scan[tx] + original[tx];
    }
}

__global__ void applyHistogram(
    float* yuv_input,
    float* cdf_input,
    unsigned char* rgb_output,
    int width,
    int height
)
{
    int tx = threadIdx.y * blockDim.x + threadIdx.x;
    int numThreadsPerBlock = blockDim.x * blockDim.y;
    int globalIdx = blockIdx.x * numThreadsPerBlock + tx;
    int totalThreads = gridDim.x * numThreadsPerBlock;
    int imgSz = width * height;

    // Load CDF values into shared memory for fast repeated access
    __shared__ float cdf[HISTOGRAM_BINS];
    for (int i = tx; i < HISTOGRAM_BINS; i += numThreadsPerBlock)
    {
        cdf[i] = cdf_input[i];
    }
    __syncthreads();

    // cdfMin is the CDF value of the lowest intensity bin
    float cdfMin = cdf[0];

    // Each thread processes pixels with stride
    for (int i = globalIdx; i < imgSz; i += totalThreads)
    {
        // Read Y value and UV components from YUV buffer
        float y = yuv_input[i];
        float u = yuv_input[i + imgSz] - 128.0f;
        float v = yuv_input[i + 2 * imgSz] - 128.0f;

        // Get the histogram bin for this pixel's Y value
        int bin = (int)(y + 0.5f);
        bin = min(max(bin, 0), HISTOGRAM_BINS - 1);

        // Apply histogram equalization formula (Eq. 1 in assignment)
        float equalized_y = 255.0f * (cdf[bin] - cdfMin) / (1.0f - cdfMin);
        equalized_y = fminf(fmaxf(equalized_y, 0.0f), 255.0f);

        // Convert equalized Y'UV back to RGB using Eq. 3
        float r = equalized_y + 1.402f * v;
        float g = equalized_y - 0.344f * u - 0.714f * v;
        float b = equalized_y + 1.772f * u;

        // Clamp and store RGB output (non-interleaved layout)
        rgb_output[i] = (unsigned char)fminf(fmaxf(r, 0.0f), 255.0f);
        rgb_output[i + imgSz] = (unsigned char)fminf(fmaxf(g, 0.0f), 255.0f);
        rgb_output[i + 2 * imgSz] = (unsigned char)fminf(fmaxf(b, 0.0f), 255.0f);
    }
}
////////////////////////////////////////////////////////////////////////////////
// Host interface to GPU histogram
////////////////////////////////////////////////////////////////////////////////

extern "C" void histogram256(
    uint* d_Histogram,
    float* d_HistogramCdf,
    void* d_DataIn,
    void* d_DataYUV,
    void* d_DataOut,
    uint imgWidth,
    uint imgHeight,
    uint imgChannels
)
{
    // Step 1: Compute histogram and YUV conversion
    dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 DimGrid(HISTOGRAM_BINS, 1, 1);

    // Zero the histogram buffer before atomicAdd accumulation
    cudaMemset(d_Histogram, 0, HISTOGRAM_BINS * sizeof(uint));

    //launch kernel of histogram
    histogram << <DimGrid, DimBlock >> > (
        (unsigned char*)d_DataIn,
        d_Histogram,
        (float*)d_DataYUV,
        imgWidth,
        imgHeight
        );
    getLastCudaError("Histogramming failed\n");
    cudaDeviceSynchronize();

    // Step 2: Compute CDF and find minimum CDF
    //launch kernel of cdf scan
    computeCDFAndMin << <DimGrid, DimBlock >> > (
        d_Histogram,
        d_HistogramCdf,
        imgWidth,
        imgHeight
        );
    getLastCudaError("CDF computation failed\n");
    cudaDeviceSynchronize();

    // Step 3: Apply histogram equalization
    //launch kernel of applying histogram equalization
    applyHistogram << <DimGrid, DimBlock >> > (
        (float*)d_DataYUV,
        d_HistogramCdf,
        (unsigned char*)d_DataOut,
        imgWidth,
        imgHeight
		);
    getLastCudaError("Histogram equalization failed\n");
    cudaDeviceSynchronize();
}




