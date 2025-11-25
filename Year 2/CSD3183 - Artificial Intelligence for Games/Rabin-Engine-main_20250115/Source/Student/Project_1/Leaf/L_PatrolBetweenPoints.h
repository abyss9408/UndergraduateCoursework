#pragma once
#include "BehaviorNode.h"
#include "Misc/NiceTypes.h"

class L_PatrolBetweenPoints : public BaseNode<L_PatrolBetweenPoints>
{
public:
    L_PatrolBetweenPoints();
protected:
    std::vector<Vec3> patrolPoints;
    size_t currentTargetIndex;
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
};