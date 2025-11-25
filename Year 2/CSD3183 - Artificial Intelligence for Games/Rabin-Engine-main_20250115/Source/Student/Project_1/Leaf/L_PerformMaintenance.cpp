#include <pch.h>
#include "L_PerformMaintenance.h"
#include "Agent/BehaviorAgent.h"

L_PerformMaintenance::L_PerformMaintenance() : maintenanceTime(0.0f), currentTime(0.0f), atWorkPosition(false), isCalibrating(false), calibrationTimer(0.0f)
{
    workPosition = Vec3(0, 0, 0); // Will be set when terrain is available
}

void L_PerformMaintenance::on_enter()
{
    maintenanceTime = RNG::range(5.0f, 12.0f); // Longer maintenance sessions
    currentTime = 0.0f;
    atWorkPosition = false;
    isCalibrating = false;
    calibrationTimer = 0.0f;

	// Set work position to robot/containment area (red squares) - top left corner of the terrain
    if (terrain)
    {
        workPosition = terrain->get_world_position(9, 9);
    }

    BehaviorNode::on_leaf_enter();
}

void L_PerformMaintenance::on_update(float dt)
{
    if (!atWorkPosition)
    {
        // Move to work position
        atWorkPosition = agent->move_toward_point(workPosition, dt);
        if (atWorkPosition)
        {
            // Face the work area (simulate working on equipment)
            Vec3 direction = workPosition - agent->get_position();
            float yaw = std::atan2(direction.x, direction.z);
            agent->set_yaw(yaw);
        }
    }
    else
    {
        currentTime += dt;

        // Simulate technical work with occasional calibration adjustments
        if (!isCalibrating && RNG::range(0.0f, 1.0f) < 0.05f) // 5% chance per frame to start calibrating
        {
            isCalibrating = true;
            calibrationTimer = RNG::range(1.0f, 3.0f);
        }

        if (isCalibrating)
        {
            calibrationTimer -= dt;

            // Simulate small movements during calibration
            if (RNG::range(0.0f, 1.0f) < 0.1f)
            {
                float smallRotation = RNG::range(-0.2f, 0.2f);
                agent->set_yaw(agent->get_yaw() + smallRotation);
            }

            if (calibrationTimer <= 0.0f)
            {
                isCalibrating = false;
            }
        }

        if (currentTime >= maintenanceTime)
        {
            on_success();
        }
    }

    display_leaf_text();
}