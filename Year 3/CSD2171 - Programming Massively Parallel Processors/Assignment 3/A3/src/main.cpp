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
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

// Utility and system includes
#include <helper_cuda.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples

// project include
#include "histogram_common.h"
#include <stdint.h>
#include "bmp.h"

constexpr float epsilon = 0.00001f;
const static char* sSDKsample = "[histogram equalization]\0";

void writeCDFToFile(const char* fileName, float* histoCdf, int count)
{
	std::ofstream file(fileName, std::ios::out | std::ios::binary);
	for (int i = 0; i < count; i += 4)
		file << histoCdf[i] << " " << histoCdf[i + 1] << " " << histoCdf[i + 2] << " " << histoCdf[i + 3] << std::endl;;
	file.close();
}

void writeHistoBinToFile(const char* fileName, unsigned int* histoBin, int count)
{
	std::ofstream file(fileName, std::ios::out | std::ios::binary);
	for (int i = 0; i < count; i += 4)
		file << histoBin[i] << " " << histoBin[i + 1] << " " << histoBin[i + 2] << " " << histoBin[i + 3] << std::endl;;
	file.close();
}

int main(int argc, char** argv)
{
	uint* h_HistogramCPU;
	uint* h_HistogramGPU;
	uchar* d_Data;
	uchar* d_DataOut;
	float* d_DataYUV;
	uint* d_Histogram;
	float* d_HistogramsCdf;
	float* h_HistogramGPUCdf;
	float* h_HistogramCPUCdf;
	StopWatchInterface* hTimer = NULL;
	int PassFailFlag = 1;

	cudaDeviceProp deviceProp;
	deviceProp.major = 0;
	deviceProp.minor = 0;

	if (argc != 7) {
		printf
		("Usage: histogram <InFile> <CPU Output File> <GPU Output File> <Histogram CPU Output File> <Histogram GPU Output File> <cdf Output File> \n\n");
		exit(0);
	}

	// set logfile name and start logs
	printf("[%s] - Starting...\n", sSDKsample);

	//Use command-line specified CUDA device, otherwise use device with highest Gflops/s
	int dev = findCudaDevice(argc, (const char**)argv);

	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

	printf("CUDA device [%s] has %d Multi-Processors, Compute %d.%d\n",
		deviceProp.name, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

	sdkCreateTimer(&hTimer);

	printf("Initializing data...\n");
	printf("...reading input data\n");
	printf("...allocating CPU memory.\n");

	Image_t* imgHandle = new Image_t;

	bmp_header header;
	bmp_read(argv[1], &header, &imgHandle->data);
	uchar* h_Data = imgHandle->data;
	imgHandle->width = header.width;
	imgHandle->height = header.height;
	imgHandle->channels = 3;
	int imageWidth = header.width;
	int imageHeight = header.height;
	int imageChannels = 3;


	int colorDepth = 255;
	double dAvgSecs;
	uint byteCount = imageWidth * imageHeight * imageChannels;

	uchar* output_image = (uchar*)malloc(sizeof(uchar) * byteCount);
	float* output_yuv = (float*)malloc(sizeof(float) * byteCount);

	h_HistogramGPU = (uint*)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));
	h_HistogramGPUCdf = (float*)malloc(HISTOGRAM256_BIN_COUNT * sizeof(float));

	printf("...allocating GPU memory and copying input data\n\n");
	checkCudaErrors(cudaMalloc((void**)&d_Data, byteCount));
	checkCudaErrors(cudaMalloc((void**)&d_DataOut, byteCount));
	checkCudaErrors(cudaMalloc((void**)&d_DataYUV, sizeof(float) * byteCount));
	checkCudaErrors(cudaMalloc((void**)&d_Histogram, HISTOGRAM256_BIN_COUNT * sizeof(uint)));
	checkCudaErrors(cudaMemcpy(d_Data, h_Data, byteCount, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&d_HistogramsCdf, HISTOGRAM256_BIN_COUNT * sizeof(float)));

	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

	histogram256(d_Histogram, d_HistogramsCdf, d_Data, d_DataYUV, d_DataOut,
		imageWidth, imageHeight, imageChannels);

	checkCudaErrors(cudaMemcpy(output_image, d_DataOut, byteCount, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(output_yuv, d_DataYUV, sizeof(float) * byteCount, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_HistogramGPU, d_Histogram, HISTOGRAM256_BIN_COUNT * sizeof(uint), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_HistogramGPUCdf, d_HistogramsCdf, HISTOGRAM256_BIN_COUNT * sizeof(float), cudaMemcpyDeviceToHost));

	sdkStopTimer(&hTimer);

	dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer);
	printf("histogram256() time: %.5f sec, %.4f MB/sec\n\n", dAvgSecs, ((double)byteCount * 1.0e-6) / dAvgSecs);
	printf("histogram256, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes, NumDevsUsed = %u\n",
		(1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs, byteCount, 1);

	printf("Shutting down...\n");

	checkCudaErrors(cudaFree(d_HistogramsCdf));
	checkCudaErrors(cudaFree(d_Histogram));
	checkCudaErrors(cudaFree(d_Data));
	checkCudaErrors(cudaFree(d_DataYUV));
	checkCudaErrors(cudaFree(d_DataOut));

	Image_t* imgOutput = new Image_t;
	imgOutput->width = imageWidth;
	imgOutput->height = imageHeight;
	imgOutput->channels = imageChannels;
	imgOutput->data = output_image;

	cudaDeviceReset();

	h_HistogramCPU = (uint*)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));
	h_HistogramCPUCdf = (float*)malloc(HISTOGRAM256_BIN_COUNT * sizeof(float));
	uchar* output_imageCPU = (uchar*)malloc(sizeof(uchar) * byteCount);
	float* h_YUV = (float*)malloc(sizeof(float) * byteCount);

	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

	histogram256CPU(
		h_HistogramCPU,
		h_HistogramCPUCdf,
		h_Data,
		h_YUV,
		output_imageCPU,
		imageWidth,
		imageHeight,
		imageChannels
	);

	sdkStopTimer(&hTimer);
	dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer); // (double)numRuns;
	printf("histogram256CPU() time: %.5f sec, %.4f MB/sec\n\n", dAvgSecs, ((double)byteCount * 1.0e-6) / dAvgSecs);
	printf("histogram256CPU, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes\n",
		(1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs, byteCount);
	printf("Shutting down...\n");

	sdkDeleteTimer(&hTimer);

	printf(" ...comparing the results\n");
	PassFailFlag = 1;
	for (uint i = 0; i < HISTOGRAM256_BIN_COUNT; i++)
		if (h_HistogramGPU[i] != h_HistogramCPU[i])
		{
			PassFailFlag = 0;
			break;
		}
	printf(PassFailFlag ? " ...CPU and GPU histogram equalization results match\n\n" : " ***CPU and GPU histogram equalization results do not match!!!***\n\n");

	//if (PassFailFlag)
	{
		writeHistoBinToFile(argv[4], h_HistogramCPU, 256);
		writeHistoBinToFile(argv[5], h_HistogramGPU, 256);
	}

	for (unsigned int i = 0; i < HISTOGRAM256_BIN_COUNT; i++)
		if (abs(h_HistogramGPUCdf[i] - h_HistogramCPUCdf[i]) > epsilon)
		{
			PassFailFlag = 0;
			break;
		}
	printf(PassFailFlag ? " ...CPU and GPU CDF output results match\n\n" : " ***CPU and GPU CDF output results do not match!!!***\n\n");

	if (PassFailFlag)
	{
		writeCDFToFile(argv[6], h_HistogramCPUCdf, 256);
	}

	for (int i = 0; i < imageWidth * imageHeight; i++)
		if (abs(output_yuv[i] - h_YUV[i]) > epsilon)
		{
			PassFailFlag = 0;
			break;
		}

	printf(PassFailFlag ? " ...CPU and GPU YUV output results match\n\n" : " ***CPU and GPU YUV output results do not match!!!***\n\n");

	for (int i = 0; i < imageWidth * imageHeight; i++)
		if (abs(output_imageCPU[i] - output_image[i]) > 3)
		{
			PassFailFlag = 0;
			break;
		}

	printf(PassFailFlag ? " ...CPU and GPU output results match\n\n" : " ***CPU and GPU output results do not match!!!***\n\n");


	//GPU results
	bmp_write(argv[3], &header, output_image);//free inside bmp_write


	imgOutput->channels = 3;
	imgOutput->width = imageWidth;
	imgOutput->height = imageHeight;
	imgOutput->data = output_imageCPU;

	//CPU results
	bmp_write(argv[2], &header, output_imageCPU);


	delete imgOutput;
	free(h_HistogramCPUCdf);
	free(h_HistogramGPUCdf);

	free(h_HistogramGPU);
	free(h_HistogramCPU);
	free(h_Data);
	free(h_YUV);

	printf("%s - Test Summary\n", sSDKsample);
	if (!PassFailFlag)
	{
		printf("Test failed!\n");
		return -1;
	}

	printf("Test passed\n");

	return 0;
}