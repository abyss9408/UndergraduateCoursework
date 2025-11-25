#include <pch.h>
#include "C_PrioritySelector.h"

C_PrioritySelector::C_PrioritySelector() : currentIndex(0) {}

void C_PrioritySelector::on_enter()
{
    currentIndex = 0;
    BehaviorNode::on_enter();
}

void C_PrioritySelector::on_update(float dt)
{
    // Always try higher priority nodes first
    for (size_t i = 0; i < children.size(); ++i)
    {
        if (i < currentIndex && children[i]->is_ready())
        {
            // Higher priority node became available, switch to it
            children[currentIndex]->set_status(NodeStatus::SUSPENDED);
            currentIndex = i;
        }
    }

    BehaviorNode* currentNode = children[currentIndex];
    currentNode->tick(dt);

    if (currentNode->succeeded())
    {
        on_success();
    }
    else if (currentNode->failed())
    {
        ++currentIndex;
        if (currentIndex >= children.size())
        {
            on_failure();
        }
    }
}