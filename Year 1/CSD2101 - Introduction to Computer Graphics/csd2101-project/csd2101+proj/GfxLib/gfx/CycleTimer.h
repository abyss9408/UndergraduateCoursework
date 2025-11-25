/*!
@file     CycleTimer.h
@author  Prasanna Ghali       (pghali@digipen.edu)

CVS: $Id: CycleTimer.h,v 1.2 2005/02/16 21:15:41 jmp Exp $

All content (c) 2002 DigiPen Institute of Technology, all rights reserved.
*//*__________________________________________________________________________*/

/*                                                                      guard
----------------------------------------------------------------------------- */

#ifndef GFX_CYCLETIMER_H_
#define GFX_CYCLETIMER_H_

/*                                                                   includes
----------------------------------------------------------------------------- */

#include "GFXInternal.h"
#if 0
/*  _________________________________________________________________________ */
class gfxCycleTimer
/*!
*/
{
  public:
    // initialization
    static void Init(void);
  
    // timing
    static void IntervalBegin(void);
    static void IntervalEnd(void);  
    
    // reporting
    static unsigned int     GetOverhead(void){return smOverhead; }
    static unsigned __int64 GetInterval(void){return smLastInterval; }
    
  private:
    static unsigned int      smOverhead;      //!< Timing overhead.
    static unsigned __int64  smLastInterval;  //!< Time of last measured interval.
    
    static unsigned int  smCyclesLo1;  //!< Low 32 bits of interval begin stamp.
    static unsigned int  smCyclesHi1;  //!< High 32 bits of interval begin stamp.
    static unsigned int  smCyclesLo2;  //!< Low 32 bits of interval end stamp.
    static unsigned int  smCyclesHi2;  //!< High 32 bits of interval end stamp.
};
#else
/*                                                                    classes
----------------------------------------------------------------------------- */

/*  _________________________________________________________________________ */
class gfxCycleTimer
/*!
*/
{
  public:
    // initialization
    static void Init(void);
  
    // timing
    static void IntervalBegin(void);
    static void IntervalEnd(void);  
    
    // reporting
    static unsigned int     GetOverhead(void);// {return smOverhead; }
    static unsigned __int64 GetInterval(void);// {return smLastInterval; }
    
  private:
    static unsigned int      smOverhead;      //!< Timing overhead.
    static unsigned __int64  smLastInterval;  //!< Time of last measured interval.
    static unsigned __int64 smCyclesBegin;   // Timestamp counter at interval begin
    static unsigned __int64 smCyclesEnd;     // Timestamp counter at interval end
    static int cpuInfo[4];  // Array to hold CPUID result
    static void CalculateOverhead();
};
#endif

#endif  /* GFX_CYCLETIMER_H_ */