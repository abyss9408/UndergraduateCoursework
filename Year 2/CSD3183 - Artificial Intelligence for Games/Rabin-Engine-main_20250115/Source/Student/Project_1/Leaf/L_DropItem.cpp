#include <pch.h>
#include "L_DropItem.h"
#include "Agent/BehaviorAgent.h"

L_DropItem::L_DropItem() : hasDropped(false) {}

void L_DropItem::on_enter()
{
    hasDropped = false;
    BehaviorNode::on_leaf_enter();
}

void L_DropItem::on_update(float dt)
{
    if (!hasDropped)
    {
        // Random chance to drop item (malfunction behavior)
        if (RNG::range(0.0f, 1.0f) < 0.3f) // 30% chance per update
        {
            hasDropped = true;
            // Store drop location in blackboard
            auto& bb = agent->get_blackboard();
            bb.set_value("Drop Location", agent->get_position());
            on_success();
        }
        else
        {
            on_failure(); // Didn't drop this time
        }
    }

    display_leaf_text();
}