#pragma once

// Include all node headers in this file

// Example Control Flow Nodes
#include "ControlFlow/C_ParallelSequencer.h"
#include "ControlFlow/C_RandomSelector.h"
#include "ControlFlow/C_Selector.h"
#include "ControlFlow/C_Sequencer.h"

// Student Control Flow Nodes
#include "ControlFlow/C_PrioritySelector.h"
#include "ControlFlow/C_WeightedSelector.h"

// Example Decorator Nodes
#include "Decorator/D_Delay.h"
#include "Decorator/D_InvertedRepeater.h"
#include "Decorator/D_RepeatFourTimes.h"

// Student Decorator Nodes
#include "Decorator/D_Cooldown.h"
#include "Decorator/D_ConditionalRepeater.h"

// Example Leaf Nodes
#include "Leaf/L_CheckMouseClick.h"
#include "Leaf/L_Idle.h"
#include "Leaf/L_MoveToFurthestAgent.h"
#include "Leaf/L_MoveToMouseClick.h"
#include "Leaf/L_MoveToRandomPosition.h"
#include "Leaf/L_PlaySound.h"

// Student Leaf Nodes
#include "Leaf/L_PatrolBetweenPoints.h"
#include "Leaf/L_InteractWithTerminal.h"
#include "Leaf/L_PrepareSample.h"
#include "Leaf/L_CheckContainment.h"
#include "Leaf/L_PerformMaintenance.h"
#include "Leaf/L_DropItem.h"
#include "Leaf/L_InteractWithNearbyAgent.h"