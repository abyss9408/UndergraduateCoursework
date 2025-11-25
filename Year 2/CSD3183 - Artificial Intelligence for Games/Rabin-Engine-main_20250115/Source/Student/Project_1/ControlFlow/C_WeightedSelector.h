#pragma once
#include "BehaviorNode.h"

class C_WeightedSelector : public BaseNode<C_WeightedSelector>
{
public:
    C_WeightedSelector();
protected:
    size_t selectedIndex;
    std::vector<float> weights;
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
    void choose_weighted_node();
};