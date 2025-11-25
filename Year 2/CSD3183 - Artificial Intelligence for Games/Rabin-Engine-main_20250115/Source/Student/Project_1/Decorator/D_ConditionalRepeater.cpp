#include <pch.h>
#include "D_ConditionalRepeater.h"

D_ConditionalRepeater::D_ConditionalRepeater() : maxRepeats(5), currentRepeats(0), shouldContinue(true) {}

void D_ConditionalRepeater::on_enter()
{
    currentRepeats = 0;
    shouldContinue = true;
    BehaviorNode::on_enter();
}

void D_ConditionalRepeater::on_update(float dt)
{
    if (!check_condition() || currentRepeats >= maxRepeats)
    {
        on_success();
        return;
    }

    BehaviorNode* child = children.front();
    child->tick(dt);

    if (child->succeeded())
    {
        currentRepeats++;
        child->set_status(NodeStatus::READY); // Reset for next iteration
    }
    else if (child->failed())
    {
        on_failure();
    }
}

bool D_ConditionalRepeater::check_condition()
{
    // Default implementation - override in specific use cases
    return currentRepeats < maxRepeats;
}