#pragma once
#include "BehaviorNode.h"
#include "Misc/NiceTypes.h"

class L_PrepareSample : public BaseNode<L_PrepareSample>
{
public:
    L_PrepareSample();
protected:
    float preparationTime;
    float currentTime;
    Vec3 labBenchPosition;
    bool atBench;
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
};