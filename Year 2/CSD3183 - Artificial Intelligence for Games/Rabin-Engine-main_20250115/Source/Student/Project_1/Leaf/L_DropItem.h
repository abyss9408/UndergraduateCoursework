#pragma once
#include "BehaviorNode.h"

class L_DropItem : public BaseNode<L_DropItem>
{
public:
    L_DropItem();
protected:
    bool hasDropped;
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
};
