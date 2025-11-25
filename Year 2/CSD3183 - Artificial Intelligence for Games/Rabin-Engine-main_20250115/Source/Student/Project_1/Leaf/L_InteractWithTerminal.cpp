#include <pch.h>
#include "L_InteractWithTerminal.h"
#include "Agent/BehaviorAgent.h"

L_InteractWithTerminal::L_InteractWithTerminal() : interactionTime(0.0f), currentTime(0.0f), atTerminal(false)
{
    terminalPosition = Vec3(0, 0, 0); // Will be set when terrain is available
}

void L_InteractWithTerminal::on_enter()
{
    interactionTime = RNG::range(2.0f, 5.0f);
    currentTime = 0.0f;
    atTerminal = false;

    // Set terminal position when terrain is available
    if (terrain)
    {
        terminalPosition = terrain->get_world_position(2, 2); // Terminal area (blue squares) - top-left
    }

    BehaviorNode::on_leaf_enter();
}

void L_InteractWithTerminal::on_update(float dt)
{
    if (!atTerminal)
    {
        atTerminal = agent->move_toward_point(terminalPosition, dt);
        if (atTerminal)
        {
            // Face the terminal
            Vec3 direction = terminalPosition - agent->get_position();
            float yaw = std::atan2(direction.x, direction.z);
            agent->set_yaw(yaw);
        }
    }
    else
    {
        currentTime += dt;
        if (currentTime >= interactionTime)
        {
            on_success();
        }
    }

    display_leaf_text();
}