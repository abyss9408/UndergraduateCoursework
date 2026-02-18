/*
 * Copyright 2026 Digipen.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef HISTOGRAM_COMMON_H
#define HISTOGRAM_COMMON_H

////////////////////////////////////////////////////////////////////////////////
// Common definitions
////////////////////////////////////////////////////////////////////////////////
#define HISTOGRAM256_BIN_COUNT 256
#define UINT_BITS 32
typedef unsigned int uint;
typedef unsigned char uchar;

#define mymin(a, b) (((a) < (b)) ? (a) : (b)) 
#define mymax(a, b) (((a) > (b)) ? (a) : (b)) 

//#define PROBABILITY(x, width, height) (x) / ((width) * (height))
#define CLAMP(x, start, end) (mymin(mymax((x), (start)), (end)))
//#define CORRECT_COLOR(cdfVal, cdfMin)  CLAMP(255 * ((cdfVal) - (cdfMin)) / (1 - (cdfMin)), 0.0, 255.0)


////////////////////////////////////////////////////////////////////////////////
// GPU-specific common definitions
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Reference CPU histogram
////////////////////////////////////////////////////////////////////////////////
extern "C" void histogram256CPU(
	uint* histo,
	float* histoCdf,
	uchar* inRGB,
	float* dataYUV,
	uchar* outRGB,
	uint imgWidth,
	uint imgHeight,
	uint imgChannels
);

////////////////////////////////////////////////////////////////////////////////
// GPU histogram
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
);
#endif
