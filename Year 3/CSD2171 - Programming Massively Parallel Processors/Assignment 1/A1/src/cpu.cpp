/* Start Header *****************************************************************/
/*!
	\file cpu.cpp

	\author Bryan Ang Wei Ze, bryanweize.ang, 2301397

	\par bryanweize.ang\@digipen.edu

	\date January 07, 2026

	\brief Copyright (C) 2026 DigiPen Institute of Technology.

	Reproduction or disclosure of this file or its contents without the prior written consent of DigiPen Institute of Technology is prohibited.
*/
/* End Header *******************************************************************/
#include "heat.h"
#include <stdio.h>
extern "C" void initPoints(
	float *pointIn,
	float *pointOut,
	uint nRowPoints
)
{
	for (uint i = 0; i < nRowPoints; ++i)
	{
		for (uint j = 0; j < nRowPoints; ++j)
		{
			// Edge points
			if (0 == i || nRowPoints - 1 == i || 0 == j || nRowPoints - 1 == j)
			{
				*(pointIn + i * nRowPoints + j) = *(pointOut + i * nRowPoints + j) = 26.67f;
				// Points (0, 10) to (0, 30) inclusive
				if (0 == i && j >= 10 && j <= 30)
				{
					*(pointIn + i * nRowPoints + j) = *(pointOut + i * nRowPoints + j) = 65.56f;
				}
			}
			// Internal points
			else
			{
				*(pointIn + i * nRowPoints + j) = *(pointOut + i * nRowPoints + j) = 0.f;
			}
		}
	}
}

extern "C" void heatDistrCPU(
	float *pointIn,
	float *pointOut,
	uint nRowPoints,
	uint nIter
)
{
	for (uint k = 0; k < nIter; ++k)
	{
		for (uint i = 1; i < nRowPoints - 1; ++i)
		{
			for (uint j = 1; j < nRowPoints - 1; ++j)
			{
				*(pointOut + i * nRowPoints + j) = (*(pointIn + (i - 1) * nRowPoints + j) +
					*(pointIn + (i + 1) * nRowPoints + j) +
					*(pointIn + i * nRowPoints + (j - 1)) +
					*(pointIn + i * nRowPoints + (j + 1))) * 0.25f;
			}
		}

		// Update pointIn for the next iteration
		for (uint it = 0; it < nRowPoints * nRowPoints; ++it)
		{
			*(pointIn + it) = *(pointOut + it);
		}
	}
}
