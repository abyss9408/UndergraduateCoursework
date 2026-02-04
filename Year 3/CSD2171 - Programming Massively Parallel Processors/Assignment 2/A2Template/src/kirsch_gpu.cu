/* Start Header *****************************************************************/
/*!
	\file kirsch_gpu.cu

	\author Bryan Ang Wei Ze, bryanweize.ang, 2301397

	\par bryanweize.ang\@digipen.edu

	\date January 31, 2026

	\brief Copyright (C) 2026 DigiPen Institute of Technology.

	Reproduction or disclosure of this file or its contents without the prior written consent of DigiPen Institute of Technology is prohibited.
*/
/* End Header *******************************************************************/

#include <helper_cuda.h>
#include "edge.h"

#define Mask_width 3
#define Mask_radius (int)(Mask_width / 2)
#define TILE_WIDTH 16
#define w (TILE_WIDTH + Mask_width - 1)

#define O_TILE_WIDTH 14
#define BLOCK_WIDTH (O_TILE_WIDTH + Mask_width - 1)

#define clamp(x) (mymin(mymax((x), 0), 255))

#ifdef CONSTANT_MEMORY
__constant__ int kirschMask[8 * 3 * 3] =
{

	5, 5, 5 ,
	-3, 0, -3 ,           /*rotation 1 */
	-3, -3, -3 ,
	5, 5, -3 ,
	5, 0, -3 ,            /*rotation 2 */
	-3, -3, -3 ,
	5, -3, -3 ,
	5, 0, -3 ,            /*rotation 3 */
	5, -3, -3 ,
	-3, -3, -3,
	5, 0, -3 ,            /*rotation 4 */
	5, 5, -3 ,
	-3, -3, -3 ,
	-3, 0, -3 ,           /*rotation 5 */
	5, 5, 5 ,
	-3, -3, -3 ,
	-3, 0, 5 ,            /*rotation 6 */
	-3, 5, 5 ,
	-3, -3, 5 ,
	-3, 0, 5 ,            /*rotation 7 */
	-3, -3, 5 ,
	-3, 5, 5 ,
	-3, 0, 5 ,            /*rotation 8 */
	-3, -3, -3
};
#endif

//Use of const  __restrict__ qualifiers for the mask parameter 
//informs the compiler that it is eligible for constant caching

///============================================================================
/// Design Option 1: Thread block size matches output tile size
///============================================================================
/// - Thread block: TILE_WIDTH x TILE_WIDTH (16x16 = 256 threads)
/// - Output tile:  TILE_WIDTH x TILE_WIDTH (16x16 = 256 pixels)
/// - Shared memory: w x w where w = TILE_WIDTH + Mask_width - 1 (18x18 = 324)
/// - Multi-stage loading: Since w*w (324) > TILE_WIDTH*TILE_WIDTH (256),
///   each thread loads multiple elements using k-th stage offset
///============================================================================

__global__ void convolution(unsigned char* I,
	const int* __restrict__ M,
	unsigned char* P,
	int channels,
	int width,
	int height)
{
	// Shared memory for input tile with halo (w x w)
	__shared__ unsigned char Ns[w][w];

	// Block and thread indices (notation from slides)
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Channel (color layer) handled by this block (0=Blue, 1=Green, 2=Red)
	int layer = blockIdx.z;
	int size = width * height;

	// Output pixel coordinates for this thread
	int out_col = bx * TILE_WIDTH + tx;
	int out_row = by * TILE_WIDTH + ty;

	//==========================================================================
	// Multi-stage tile loading into shared memory
	// Total elements: w * w = 324
	// Threads per block: TILE_WIDTH * TILE_WIDTH = 256
	// Stages needed: ceil(324/256) = 2 (since w*w < 2*TILE_WIDTH*TILE_WIDTH)
	//==========================================================================
	for (int k = 0; k < 2; ++k)
	{
		// Flattened 1D index for k-th stage loading
		int index_Ns_1D = ty * TILE_WIDTH + tx + k * TILE_WIDTH * TILE_WIDTH;

		// Only load if within shared memory bounds
		if (index_Ns_1D < w * w)
		{
			// Convert 1D index to 2D shared memory coordinates
			int index_Ns_x = index_Ns_1D % w;
			int index_Ns_y = index_Ns_1D / w;

			// Compute global memory coordinates
			// Shift by -Mask_radius to account for halo region
			int sX = bx * TILE_WIDTH + index_Ns_x - Mask_radius;
			int sY = by * TILE_WIDTH + index_Ns_y - Mask_radius;

			// Boundary check - use 0 for ghost cells outside image
			if (sX >= 0 && sX < width && sY >= 0 && sY < height)
			{
				Ns[index_Ns_y][index_Ns_x] = I[sY * width + sX + layer * size];
			}
			else
			{
				Ns[index_Ns_y][index_Ns_x] = 0;
			}
		}
	}

	// Synchronize to ensure all shared memory is loaded
	__syncthreads();

	//==========================================================================
	// Convolution computation
	// Apply all 8 Kirsch masks and find maximum response
	//==========================================================================
	if (out_row < height && out_col < width)
	{
		int max_sum = 0;

		// Apply all 8 Kirsch masks
		for (int m = 0; m < 8; ++m)
		{
			int sum = 0;

			// 3x3 convolution with mask m
			for (int j = 0; j < Mask_width; ++j)
			{
				for (int i = 0; i < Mask_width; ++i)
				{
					// Mask value from constant memory
					int mask_val = kirschMask[m * 9 + j * Mask_width + i];

					// Pixel value from shared memory
					// Thread (tx, ty) accesses Ns[ty+j][tx+i] for its 3x3 neighborhood
					int pixel_val = (int)Ns[ty + j][tx + i];
					sum += mask_val * pixel_val;
				}
			}

			// Keep maximum convolution response across all 8 masks
			if (sum > max_sum)
			{
				max_sum = sum;
			}
		}

		// Divide by 8 and clamp result to [0, 255]
		int result = max_sum / 8;
		if (result < 0) result = 0;
		if (result > 255) result = 255;

		// Write to output (non-interleaved: all B, then all G, then all R)
		P[out_row * width + out_col + layer * size] = (unsigned char)result;
	}
}

////////////////////////////////////////////////////////////////////////////////
// Host interface to GPU 
////////////////////////////////////////////////////////////////////////////////
extern "C" void kirschEdgeDetectorGPU(
	void* d_ImgDataIn,
	void* d_ImgDataOut,
	unsigned imgChannels,
	unsigned imgWidth,
	unsigned imgHeight
)
{

	dim3 dimGrid((imgWidth - 1) / TILE_WIDTH + 1,
		(imgHeight - 1) / TILE_WIDTH + 1, 3);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

	convolution << <dimGrid, dimBlock >> > ((unsigned char*)d_ImgDataIn,
		kirschMask,
		(unsigned char*)d_ImgDataOut,
		(int)imgChannels,
		(int)imgWidth,
		(int)imgHeight);

	getLastCudaError("Compute the kirsch edge detection failed\n");
	cudaDeviceSynchronize();
}
