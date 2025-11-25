#include <pch.h>
#include "C_WeightedSelector.h"

C_WeightedSelector::C_WeightedSelector() : selectedIndex(0)
{
    // Default weights - can be customized
    weights = { 0.4f, 0.3f, 0.2f, 0.1f }; // Higher weight = more likely to be selected
}

void C_WeightedSelector::on_enter()
{
    BehaviorNode::on_enter();
    choose_weighted_node();
}

void C_WeightedSelector::on_update(float dt)
{
    BehaviorNode* currentNode = children[selectedIndex];
    currentNode->tick(dt);

    if (currentNode->succeeded())
    {
        on_success();
    }
    else if (currentNode->failed())
    {
        // Try to find another available node
        bool foundAlternative = false;
        for (size_t i = 0; i < children.size(); ++i)
        {
            if (i != selectedIndex && children[i]->is_ready())
            {
                selectedIndex = i;
                foundAlternative = true;
                break;
            }
        }

        if (!foundAlternative)
        {
            on_failure();
        }
    }
}

void C_WeightedSelector::choose_weighted_node()
{
    float totalWeight = 0.0f;
    for (size_t i = 0; i < std::min(children.size(), weights.size()); ++i)
    {
        if (children[i]->is_ready())
        {
            totalWeight += weights[i];
        }
    }

    float random = RNG::range(0.0f, totalWeight);
    float cumulative = 0.0f;

    for (size_t i = 0; i < std::min(children.size(), weights.size()); ++i)
    {
        if (children[i]->is_ready())
        {
            cumulative += weights[i];
            if (random <= cumulative)
            {
                selectedIndex = i;
                return;
            }
        }
    }

    selectedIndex = 0; // Fallback
}