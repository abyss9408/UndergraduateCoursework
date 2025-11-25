/*!
@file    Math.h
@author  Prasanna Ghali       (pghali@digipen.edu)

CVS: $Id: Math.h,v 1.6 2005/02/16 21:15:41 jmp Exp $

All content (c) 2002 DigiPen Institute of Technology, all rights reserved.
*//*__________________________________________________________________________*/

/*                                                                      guard
----------------------------------------------------------------------------- */

#ifndef GFX_MATH_H_
#define GFX_MATH_H_

/*                                                                  constants
----------------------------------------------------------------------------- */

// pi
const float  PI = 4.0f * std::atan(1.0f);
const float  TWO_PI = 2.0f * PI;
const float  HALF_PI = 0.5f * PI;
const float  DEG_TO_RAD = PI / 180.0f;
const float  RAD_TO_DEG = 180.0f / PI;


/*                                                                  functions
----------------------------------------------------------------------------- */

/*  _________________________________________________________________________ */
template<typename T_>
inline T_ min(const T_& a, const T_& b, const T_& c)
/*! Return the minimum of three elements.

    @param a -->  1st element.
    @param b -->  2nd element.
    @param c -->  3rd element.
    
    @return
    The minimum of the three input elements.
*/
{
  return (((a < b ? a : b)) < c ? ((a < b ? a : b)) : c);
}

/*  _________________________________________________________________________ */
template<typename T_>
inline T_ max(const T_& a, const T_& b, const T_& c)
/*! Return the maximum of three elements.

    @param a -->  1st element.
    @param b -->  2nd element.
    @param c -->  3rd element.
    
    @return
    The maximum of the three input elements.
*/
{
  return (((a > b ? a : b)) > c ? ((a > b ? a : b)) : c);
}

/*  _________________________________________________________________________ */
inline float deg2rad(float d)
/*! Converts from degrees to radians

    @param d -->  The value in degrees.
    
    @return
    The value d in radians.
*/
{
  return (d * DEG_TO_RAD);
}

/*  _________________________________________________________________________ */
inline float rad2deg(float r)
/*! Converts from radians to degrees.

    @param r -->  The value in radians.
    
    @return
    The value r in degrees.
*/
{
  return (r * RAD_TO_DEG);
}

/*  _________________________________________________________________________ */
inline float fclamp(float f, float l, float h)
/*! Clamp a floating point value.

    @param f -->  The value to clamp.
    @param l -->  The minimum value.
    @param h -->  The maximum value.
    
    @return
    The value f clamped between l and h.
*/
{
  return (f < l ? l : (f > h ? h : f));
}

/*  _________________________________________________________________________ */
inline float fclampw(float f, float l, float h)
/*! Clamp a floating point value by wrapping it if it is out of range.

    @param f -->  The value to clamp.
    @param l -->  The minimum value.
    @param h -->  The maximum value.
    
    @return
    The value f clamped between l and h.
*/
{
  if(f < l)
    return (f + (h - l));
  else if(f > h)
    return (l + f - h);
  else
    return (f);
}

#if 0
/*  _________________________________________________________________________ */
inline int ff2i(float f)
/*! Fast float to int.

    @param f -->  The value to convert.
    
    @return
    The value f, converted to an integer.
*/
{
int  i;

  __asm fld   f;
  __asm fistp i;
  return (i);
}
#else
#include <cstdint>
#include <xmmintrin.h>

inline int32_t ff2i(float value) {
    return _mm_cvtt_ss2si(_mm_set_ss(value));
}

#endif

/*                                                                     macros
----------------------------------------------------------------------------- */

// 28.4 fixed-point operations
#define FP28_MUL(a,b)  (((a) * (b)) >> 4)
#define FP28_DIV(a,b)  (((a) << 4) / (b))


#endif  /* GFX_H_ */