#pragma once
#include "BehaviorNode.h"
#include "Misc/NiceTypes.h"

class L_PerformMaintenance : public BaseNode<L_PerformMaintenance>
{
public:
    L_PerformMaintenance();
protected:
    float maintenanceTime;
    float currentTime;
    Vec3 workPosition;
    bool atWorkPosition;
    bool isCalibrating;
    float calibrationTimer;
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
};