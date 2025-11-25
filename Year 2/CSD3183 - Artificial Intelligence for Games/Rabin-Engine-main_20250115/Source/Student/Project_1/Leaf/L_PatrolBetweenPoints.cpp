#include <pch.h>
#include "L_PatrolBetweenPoints.h"
#include "Agent/BehaviorAgent.h"

L_PatrolBetweenPoints::L_PatrolBetweenPoints() : currentTargetIndex(0)
{
    // Initialize empty - will be set up when terrain is available
    patrolPoints.clear();
}

void L_PatrolBetweenPoints::on_enter()
{
    // Set up patrol points when terrain is available
    if (patrolPoints.empty() && terrain)
    {
        patrolPoints.push_back(terrain->get_world_position(2, 2));    // Terminal area (blue) - top-left corner
        patrolPoints.push_back(terrain->get_world_position(2, 17));   // Lab benches (green) - top-right corner
        patrolPoints.push_back(terrain->get_world_position(17, 17)); // Robot/containment area (red) - bottom-right corner
        patrolPoints.push_back(terrain->get_world_position(17, 3));   // Security area (yellow) - bottom-left corner
    }

    BehaviorNode::on_leaf_enter();
}

void L_PatrolBetweenPoints::on_update(float dt)
{
    if (patrolPoints.empty())
    {
        on_failure();
        return;
    }

    const Vec3& targetPoint = patrolPoints[currentTargetIndex];
    bool reached = agent->move_toward_point(targetPoint, dt);

    if (reached)
    {
        currentTargetIndex = (currentTargetIndex + 1) % patrolPoints.size();

        // Patrol is never "complete" - it continues indefinitely
        if (currentTargetIndex == 0)
        {
            on_success(); // Completed one full patrol cycle
        }
    }

    display_leaf_text();
}