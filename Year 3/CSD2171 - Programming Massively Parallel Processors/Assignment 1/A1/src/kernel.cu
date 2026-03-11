/* Start Header *****************************************************************/
/*!
	\file kernel.cu

	\author Bryan Ang Wei Ze, bryanweize.ang, 2301397

	\par bryanweize.ang\@digipen.edu

	\date January 07, 2026

	\brief Copyright (C) 2026 DigiPen Institute of Technology.

	Reproduction or disclosure of this file or its contents without the prior written consent of DigiPen Institute of Technology is prohibited.
*/
/* End Header *******************************************************************/

#include <helper_cuda.h>
////////////////////////////////////////////////////////////////////

#define BLOCK_SIZE 32
typedef unsigned int uint;
__global__ void heatDistrCalc(float* in, float* out, uint nRowPoints)
{
	// Calculate global thread coordinates
	uint i = blockIdx.y * blockDim.y + threadIdx.y;
	uint j = blockIdx.x * blockDim.x + threadIdx.x;

	// Only process internal grid points
	if (i > 0 && i < nRowPoints - 1 && j > 0 && j < nRowPoints - 1)
	{
		// Calculate the index for the 1D array representation of the 2D grid
		uint idx = i * nRowPoints + j;
		// Apply the heat distribution formula
		out[idx] = 0.25f * (in[idx - nRowPoints] + // Top (i-1, j)
			in[idx + nRowPoints] + // Bottom (i+1, j)
			in[idx - 1] + // Left (i, j-1)
			in[idx + 1]); // Right (i, j+1)
	}
}


__global__ void heatDistrUpdate(float* in, float* out, uint nRowPoints)
{
	// Calculate global thread coordinates
	uint i = blockIdx.y * blockDim.y + threadIdx.y;
	uint j = blockIdx.x * blockDim.x + threadIdx.x;

	// Make sure we are within bounds
	if (i < nRowPoints && j < nRowPoints)
	{
		// Calculate the index for the 1D array representation of the 2D grid
		uint idx = i * nRowPoints + j;
		// Update the input array with the new values from the output array
		in[idx] = out[idx];
	}
}

extern "C" void heatDistrGPU(
	float* d_DataIn,
	float* d_DataOut,
	uint nRowPoints,
	uint nIter
)
{
	dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 DimGrid2(ceil(((float)nRowPoints) / BLOCK_SIZE), ceil(((float)nRowPoints) / BLOCK_SIZE), 1);

	for (uint k = 0; k < nIter; ++k) {
		heatDistrCalc << <DimGrid2, DimBlock >> > (d_DataIn, d_DataOut, nRowPoints);
		getLastCudaError("heatDistrCalc failed\n");
		cudaDeviceSynchronize();
		heatDistrUpdate << <DimGrid2, DimBlock >> > (d_DataIn, d_DataOut, nRowPoints);
		getLastCudaError("heatDistrUpdate failed\n");
		cudaDeviceSynchronize();
	}
}
