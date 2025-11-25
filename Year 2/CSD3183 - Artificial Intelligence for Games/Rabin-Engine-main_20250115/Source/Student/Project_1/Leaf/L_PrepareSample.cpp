#include <pch.h>
#include "L_PrepareSample.h"
#include "Agent/BehaviorAgent.h"

L_PrepareSample::L_PrepareSample() : preparationTime(0.0f), currentTime(0.0f), atBench(false)
{
    labBenchPosition = Vec3(0, 0, 0); // Will be set when terrain is available
}

void L_PrepareSample::on_enter()
{
    preparationTime = RNG::range(3.0f, 7.0f);
    currentTime = 0.0f;
    atBench = false;

    // Set lab bench position when terrain is available
    if (terrain)
    {
        labBenchPosition = terrain->get_world_position(2, 17); // Lab bench area (green squares) - top-right
    }

    BehaviorNode::on_leaf_enter();
}

void L_PrepareSample::on_update(float dt)
{
    if (!atBench)
    {
        atBench = agent->move_toward_point(labBenchPosition, dt);
    }
    else
    {
        currentTime += dt;
        if (currentTime >= preparationTime)
        {
            on_success();
        }
    }

    display_leaf_text();
}