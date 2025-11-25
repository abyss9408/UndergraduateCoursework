#pragma once
#include "BehaviorNode.h"

class D_ConditionalRepeater : public BaseNode<D_ConditionalRepeater>
{
public:
    D_ConditionalRepeater();
protected:
    int maxRepeats;
    int currentRepeats;
    bool shouldContinue;
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
    virtual bool check_condition(); // Override in derived classes
};