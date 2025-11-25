#pragma once
#include "BehaviorNode.h"
#include "Misc/NiceTypes.h"

class L_CheckContainment : public BaseNode<L_CheckContainment>
{
public:
    L_CheckContainment();
protected:
    float checkTime;
    float currentTime;
    Vec3 containmentPosition;
    bool atContainment;
    bool isNervous;
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
};