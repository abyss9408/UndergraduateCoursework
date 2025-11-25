#include <pch.h>
#include "L_InteractWithNearbyAgent.h"

L_InteractWithNearbyAgent::L_InteractWithNearbyAgent() : interactionTime(0.0f), currentTime(0.0f), targetAgent(nullptr), foundTarget(false), atTarget(false) {}

void L_InteractWithNearbyAgent::on_enter()
{
    interactionTime = RNG::range(2.0f, 4.0f);
    currentTime = 0.0f;
    foundTarget = false;
    atTarget = false;
    targetAgent = nullptr;

    // Find a nearby agent to interact with
    const auto& allAgents = agents->get_all_agents();
    const auto& myPos = agent->get_position();
    float closestDistance = std::numeric_limits<float>().max();

    for (auto* otherAgent : allAgents)
    {
        // Don't interact with self or camera
        if (otherAgent != agent && std::string(otherAgent->get_type()) != "Camera")
        {
            const auto& otherPos = otherAgent->get_position();
            float distance = Vec3::Distance(myPos, otherPos);

            // Find closest agent within reasonable range
            if (distance < closestDistance && distance < 50.0f)
            {
                closestDistance = distance;
                targetAgent = otherAgent;
                foundTarget = true;
            }
        }
    }

    BehaviorNode::on_leaf_enter();
}

void L_InteractWithNearbyAgent::on_update(float dt)
{
    if (!foundTarget || !targetAgent)
    {
        on_failure();
        return;
    }

    if (!atTarget)
    {
        // Move close to the target agent
        Vec3 targetPos = targetAgent->get_position();
        Vec3 myPos = agent->get_position();

        // Stop a bit away from the target to simulate conversation distance
        Vec3 direction = myPos - targetPos;
        direction.Normalize();
        Vec3 conversationPos = targetPos + (direction * 8.0f); // Stand 8 units away

        atTarget = agent->move_toward_point(conversationPos, dt);

        if (atTarget)
        {
            // Face the other agent
            Vec3 lookDirection = targetPos - agent->get_position();
            float yaw = std::atan2(lookDirection.x, lookDirection.z);
            agent->set_yaw(yaw);
        }
    }
    else
    {
        // "Talking" - just wait for interaction time
        currentTime += dt;
        if (currentTime >= interactionTime)
        {
            on_success();
        }
    }

    display_leaf_text();
}