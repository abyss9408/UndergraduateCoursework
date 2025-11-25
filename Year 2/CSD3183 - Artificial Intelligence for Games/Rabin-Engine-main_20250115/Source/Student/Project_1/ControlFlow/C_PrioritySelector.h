#pragma once
#include "BehaviorNode.h"

class C_PrioritySelector : public BaseNode<C_PrioritySelector>
{
public:
    C_PrioritySelector();
protected:
    size_t currentIndex;
    virtual void on_enter() override;
    virtual void on_update(float dt) override;;
};