#pragma once
#include "BehaviorNode.h"

class D_Cooldown : public BaseNode<D_Cooldown>
{
public:
    D_Cooldown();
protected:
    float cooldownTime;
    float currentCooldown;
    bool childCompleted;
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
};