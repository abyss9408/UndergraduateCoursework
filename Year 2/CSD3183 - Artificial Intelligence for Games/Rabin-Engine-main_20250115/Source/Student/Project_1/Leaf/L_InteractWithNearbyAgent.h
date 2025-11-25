#pragma once
#include "BehaviorNode.h"
#include "Agent/BehaviorAgent.h"

class L_InteractWithNearbyAgent : public BaseNode<L_InteractWithNearbyAgent>
{
public:
    L_InteractWithNearbyAgent();
protected:
    float interactionTime;
    float currentTime;
    Agent* targetAgent;
    bool foundTarget;
    bool atTarget;
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
};