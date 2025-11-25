#pragma once
#include "BehaviorNode.h"
#include "Misc/NiceTypes.h"

class L_InteractWithTerminal : public BaseNode<L_InteractWithTerminal>
{
public:
    L_InteractWithTerminal();
protected:
    float interactionTime;
    float currentTime;
    Vec3 terminalPosition;
    bool atTerminal;
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
};