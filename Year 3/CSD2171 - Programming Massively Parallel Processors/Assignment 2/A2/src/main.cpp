/*
* Copyright 2024 Digipen.  All rights reserved.
*
* Please refer to the end user license associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms
* is strictly prohibited.
*
*/
// CUDA Runtime
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Utility and system includes
#include <helper_cuda.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples

// project include
#include "bmp.h"
#include "edge.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

const static char *sSDKsample = "[kirsch]\0";

int main(int argc, char **argv)
{
	uchar *h_DataIn;
	uchar *cpuOutput_image;
	uchar *d_DataIn;
	uchar *d_DataOut;
	int *d_Mask;
	bmp_header header;
	StopWatchInterface *hTimer = NULL;
	int PassFailFlag = 1;
	cudaDeviceProp deviceProp;
	deviceProp.major = 0;
	deviceProp.minor = 0;

	if (argc != 4) {
		printf
		("Usage: kirsch <InFile> <CPUOutFile> <GPUOutFile>  \n\n");
		exit(0);
	}

	// set logfile name and start logs
	printf("[%s] - Starting...\n", sSDKsample);

	//Use command-line specified CUDA device, otherwise use device with highest Gflops/s
	int dev = findCudaDevice(argc, (const char **)argv);

	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

	printf("CUDA device [%s] has %d Multi-Processors, Compute %d.%d\n",
		deviceProp.name, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

	sdkCreateTimer(&hTimer);

	printf("Initializing data...\n");
	printf("...reading input data\n");
	printf("...allocating CPU memory.\n");

	bmp_read(argv[1], &header, &h_DataIn);

	int imageWidth = header.width;
	int imageHeight = header.height;
	int imageChannels = 3;

	double dAvgSecs;
	uint byteCount = imageWidth*imageHeight*imageChannels*sizeof(unsigned char);

	uchar *gpuOutput_image = (uchar *)malloc(sizeof(uchar) * byteCount);
	memcpy_s(gpuOutput_image, byteCount, h_DataIn, byteCount);

	printf("...allocating GPU memory and copying input data\n\n");
	checkCudaErrors(cudaMalloc((void **)&d_DataIn, byteCount));
	checkCudaErrors(cudaMalloc((void **)&d_DataOut, byteCount));
	checkCudaErrors(cudaMalloc((void **)&d_Mask, 8 * 9 * sizeof(int)));
	checkCudaErrors(cudaMemcpy(d_DataIn, h_DataIn, byteCount, cudaMemcpyHostToDevice));
#ifndef CONSTANT_MEMORY
	checkCudaErrors(cudaMemcpy(d_Mask, kirschFilter, 8 * 9 * sizeof(int), cudaMemcpyHostToDevice));
#endif
	checkCudaErrors(cudaDeviceSynchronize());
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
#ifdef CONSTANT_MEMORY
	kirschEdgeDetectorGPU(	d_DataIn, d_DataOut, 
							imageChannels, imageWidth, imageHeight  );
#else
	kirschEdgeDetectorGPU(d_DataIn, d_Mask, d_DataOut,
		imageChannels, imageWidth, imageHeight);
#endif
//	printf("\nValidating GPU results...\n");
//	printf(" ...reading back GPU results\n");
	checkCudaErrors(cudaMemcpy(gpuOutput_image, d_DataOut, byteCount, cudaMemcpyDeviceToHost));

	sdkStopTimer(&hTimer);

	dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer);
	printf("kirschEdgeDetectionGPU() time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs, ((double)byteCount * 1.0e-6) / dAvgSecs);
	printf("kirsch edge detection, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes, NumDevsUsed = %u\n",
		(1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs, byteCount, 1);

	printf("Shutting down GPU...\n\n");

	checkCudaErrors(cudaFree(d_Mask));
	checkCudaErrors(cudaFree(d_DataIn));
	checkCudaErrors(cudaFree(d_DataOut));
	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	cudaDeviceReset();

	cpuOutput_image = (uchar *)malloc(sizeof(uchar) * byteCount);
	memcpy_s(cpuOutput_image, byteCount, h_DataIn, byteCount);

	//printf("...kirschEdgeDetectorCPU()\n");
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

	kirschEdgeDetectorCPU(
		h_DataIn, (int *) kirschFilter, cpuOutput_image,
		imageChannels, imageWidth, imageHeight
	);
	sdkStopTimer(&hTimer);
	dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer); 
	printf("kirschEdgeDetectorCPU() time: %.5f sec, %.4f MB/sec\n\n", dAvgSecs, ((double)byteCount * 1.0e-6) / dAvgSecs);
	printf("kirschEdgeDetectorCPU, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes\n",
		(1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs, byteCount);
	printf("CPU version done...\n\n");

	sdkDeleteTimer(&hTimer);
#if 1
	printf("...comparing the results of GPU and CPU version\n");
	for (uint i = 0; i < byteCount; i++)
		if (cpuOutput_image[i] != gpuOutput_image[i])
		{
			PassFailFlag = 0;
			break;
		}
	printf(PassFailFlag ? " ...kirsch edge detection matched\n\n" : " ***kirsch edge detection do not match!!!***\n\n");
#endif
	// write data to output bmp image file
	//GPU results
	bmp_write(argv[3], &header, gpuOutput_image);//free inside bmp_write
	//CPU results
	bmp_write(argv[2], &header, cpuOutput_image);
	free(h_DataIn);	

//	printf("%s - Test Summary\n", sSDKsample);
#if 0
	if (!PassFailFlag)
	{
		printf("Test failed!\n");
		return -1;
	}

	printf("Test passed\n");
#endif

	return 0;
}
