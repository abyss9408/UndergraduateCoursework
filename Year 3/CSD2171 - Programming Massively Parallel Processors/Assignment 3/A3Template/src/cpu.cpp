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

#include <assert.h>
#include "histogram_common.h"

//assume inRGB is RGB model
// dataYUV is for YUV model 
// inRGB, dataYUV, outRGB : non-interleaving layout
// histo, histoCdf : for Y component only

extern "C" void histogram256CPU(
    uint* histo,
    float* histoCdf,
    uchar* inRGB,
    float* dataYUV,
    uchar* outRGB,
    uint imgWidth,
    uint imgHeight,
    uint imgChannels
)
{
    // Validate inputs
    assert(imgChannels >= 3);

    //initialize histogram bin and cdf values
    for (int i = 0; i < HISTOGRAM256_BIN_COUNT; i++) {
        histo[i] = 0;
        histoCdf[i] = 0.0f;
    }

    int imgSz = imgWidth * imgHeight;
    float totalPixels = (float)(imgWidth * imgHeight);

    //RGB to YUV conversion
    for (unsigned int row = 0; row < imgHeight; row++) {
        for (unsigned int col = 0; col < imgWidth; col++) {
            int i = row * imgWidth + col;

            // Convert RGB to YUV (BT.601 standard)
            float r = (float)inRGB[i];
            float g = (float)inRGB[i + imgSz];
            float b = (float)inRGB[i + 2 * imgSz];

            float y = 0.299f * r + 0.587f * g + 0.114f * b;
            float u = -0.169f * r - 0.331f * g + 0.499f * b + 128.0f;
            float v = 0.499f * r - 0.418f * g - 0.0813f * b + 128.0f;

            dataYUV[i] = CLAMP(y, 0.0f, 255.0f);
            dataYUV[i + imgSz] = CLAMP(u, 0.0f, 255.0f);
            dataYUV[i + 2 * imgSz] = CLAMP(v, 0.0f, 255.0f);
        }
    }

    //histogramming on Y component with proper binning
    for (unsigned int row = 0; row < imgHeight; row++) {
        for (unsigned int col = 0; col < imgWidth; col++) {
            int i = row * imgWidth + col;

            // Use proper rounding to nearest integer for bin index
            float y_val = dataYUV[i];
            int bin = (int)(y_val + 0.5f); // Round to nearest integer
            bin = CLAMP(bin, 0, HISTOGRAM256_BIN_COUNT - 1);
            histo[bin]++;
        }
    }

    //cdf calculation with proper probability
    histoCdf[0] = (float)histo[0] / totalPixels;
    for (int i = 1; i < HISTOGRAM256_BIN_COUNT; i++) {
        float pdf = (float)histo[i] / totalPixels;
        histoCdf[i] = pdf + histoCdf[i - 1];
    }

    float cdfMin = histoCdf[0];

    // Apply histogram equalization
    for (unsigned int row = 0; row < imgHeight; row++) {
        for (unsigned int col = 0; col < imgWidth; col++) {
            int i = row * imgWidth + col;

            // Get bin index with proper rounding
            float y_val = dataYUV[i];
            int bin = (int)(y_val + 0.5f);
            bin = CLAMP(bin, 0, HISTOGRAM256_BIN_COUNT - 1);

            // Apply histogram equalization formula
            float equalized_y;

            equalized_y = 255.0f * (histoCdf[bin] - cdfMin) / (1.0f - cdfMin);

            equalized_y = CLAMP(equalized_y, 0.0f, 255.0f);

            // Get UV components
            float u = dataYUV[i + imgSz] - 128.0f;
            float v = dataYUV[i + 2 * imgSz] - 128.0f;

            // Convert YUV back to RGB
            float r = equalized_y + 1.402f * v;
            float g = equalized_y - 0.344f * u - 0.714f * v;
            float b = equalized_y + 1.772f * u;

            // Store results
            outRGB[i] = (uchar)CLAMP(r, 0.0f, 255.0f);
            outRGB[i + imgSz] = (uchar)CLAMP(g, 0.0f, 255.0f);
            outRGB[i + 2 * imgSz] = (uchar)CLAMP(b, 0.0f, 255.0f);
        }
    }
}
