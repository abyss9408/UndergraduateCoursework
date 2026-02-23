Student Name: Bryan Ang Wei Ze

Project Name: Behavior Trees

What I implemented: A sci-fi research laboratory simulation with 4 NPCs exhibiting realistic lab behaviors using custom behavior trees. Created 10 new behavior tree nodes (2 control flow, 3 decorators, 6 leaf nodes) and 4 unique behavior trees with complex interactions between scientists, security, and technician robots.

Directions (if needed): Run the project and observe the lab simulation. The Lead Scientist works at terminal, the Lab Assistant nervously prepares samples, the Security Officer patrols, and the Robot Technician performs maintenance as well as occasionally dropping items. All NPCs will occasionally stop and interact with each other.

What I liked about the project and framework: The behavior tree system is very intuitive and modular. Easy to create complex AI behaviors by combining simple nodes. The framework provides good tools for agent movement, blackboard communication, and visual debugging. Really enjoyed seeing the emergent interactions between NPCs.

What I disliked about the project and framework: The file registration process (NodeHeaders.h, Nodes.def) is somewhat tedious and error-prone.

Any difficulties I experienced while doing the project: Initial confusion about the BaseNode template pattern and proper node cloning. It took some time to understand the lifecycle of on_enter/on_update/on_exit. I had to debug issues with agent finding and blackboard data types.

Hours spent: 12-14 hours

New selector node (name): C_PrioritySelector, C_WeightedSelector

New decorator nodes (names): D_Cooldown, D_ConditionalRepeater

11 total nodes (names): C_PrioritySelector, C_WeightedSelector, D_Cooldown, D_ConditionalRepeater, L_PatrolBetweenPoints, L_InteractWithTerminal, L_PrepareSample, L_CheckContainment, L_PerformMaintainance, L_DropItem, L_InteractWithNearbyAgent

4 Behavior trees (names): LeadScientist, LabAssistant, SecurityOfficer, RobotTechnician

Extra credit: