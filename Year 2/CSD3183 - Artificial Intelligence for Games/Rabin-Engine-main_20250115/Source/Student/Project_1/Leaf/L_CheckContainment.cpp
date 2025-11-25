#include <pch.h>
#include "L_CheckContainment.h"
#include "Agent/BehaviorAgent.h"

L_CheckContainment::L_CheckContainment() : checkTime(0.0f), currentTime(0.0f), atContainment(false), isNervous(false)
{
    containmentPosition = Vec3(0, 0, 0); // Will be set when terrain is available
}

void L_CheckContainment::on_enter()
{
    checkTime = RNG::range(1.0f, 3.0f);
    currentTime = 0.0f;
    atContainment = false;
    isNervous = RNG::coin_toss(); // Random nervousness

    // Set containment position when terrain is available
    if (terrain)
    {
        containmentPosition = terrain->get_world_position(17, 17); // Robot/containment area (red squares) - bottom-right
    }

    BehaviorNode::on_leaf_enter();
}

void L_CheckContainment::on_update(float dt)
{
    if (!atContainment)
    {
        atContainment = agent->move_toward_point(containmentPosition, dt);
    }
    else
    {
        currentTime += dt;

        // Nervous behavior - occasional quick glances away
        if (isNervous && RNG::range(0.0f, 1.0f) < 0.1f)
        {
            float randomYaw = RNG::range(0.0f, TWO_PI);
            agent->set_yaw(randomYaw);
        }

        if (currentTime >= checkTime)
        {
            on_success();
        }
    }

    display_leaf_text();
}