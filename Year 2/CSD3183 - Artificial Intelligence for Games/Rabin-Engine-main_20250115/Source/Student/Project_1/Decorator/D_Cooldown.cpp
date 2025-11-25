#include <pch.h>
#include "D_Cooldown.h"

D_Cooldown::D_Cooldown() : cooldownTime(3.0f), currentCooldown(0.0f), childCompleted(false) {}

void D_Cooldown::on_enter()
{
    if (currentCooldown <= 0.0f)
    {
        BehaviorNode::on_enter();
        childCompleted = false;
    }
    else
    {
        set_status(NodeStatus::RUNNING);
    }
}

void D_Cooldown::on_update(float dt)
{
    if (currentCooldown > 0.0f)
    {
        currentCooldown -= dt;
        if (currentCooldown <= 0.0f)
        {
            on_failure(); // Cooldown complete, but child hasn't run
        }
        return;
    }

    if (!childCompleted)
    {
        BehaviorNode* child = children.front();
        child->tick(dt);

        if (child->succeeded() || child->failed())
        {
            childCompleted = true;
            currentCooldown = cooldownTime;
            set_status(child->get_status());
            set_result(child->get_result());
        }
    }
}