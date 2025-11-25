#include <pch.h>
#include "Projects/ProjectOne.h"
#include "Agent/CameraAgent.h"

void ProjectOne::setup()
{
    // You can change properties here or at runtime from a behavior tree leaf node
    // Look in Agent.h for all of the setters, like these:
    // man->set_color(Vec3(1, 0, 1));
    // man->set_scaling(Vec3(7,7,7));
    // man->set_position(Vec3(100, 0, 100));

    // Create an agent with a different 3D model:
    // 1. (optional) Add a new 3D model to the framework other than the ones provided:
    //    A. Find a ".sdkmesh" model or use https://github.com/walbourn/contentexporter
    //       to convert fbx files (many end up corrupted in this process, so good luck!)
    //    B. Add a new AgentModel enum for your model in Agent.h (like the existing Man or Tree).
    // 2. Register the new model with the engine, so it associates the file path with the enum
    //    A. Here we are registering all of the extra models that already come in the package.
    Agent::add_model("Assets\\tree.sdkmesh", Agent::AgentModel::Tree);
    Agent::add_model("Assets\\car.sdkmesh", Agent::AgentModel::Car);
    Agent::add_model("Assets\\bird.sdkmesh", Agent::AgentModel::Bird);
    Agent::add_model("Assets\\ball.sdkmesh", Agent::AgentModel::Ball);
    Agent::add_model("Assets\\hut.sdkmesh", Agent::AgentModel::Hut);

    // You can technically load any map you want, even create your own map file,
    // but behavior agents won't actually avoid walls or anything special, unless you code
    // that yourself (that's the realm of project 2)
    terrain->goto_map(0);

    // You can also enable the pathing layer and set grid square colors as you see fit.
    // Works best with map 0, the completely blank map
    terrain->pathLayer.set_enabled(true);

    // Terminal area (blue) - Small compact area (3x3)
    for (int row = 1; row <= 3; ++row)
    {
        for (int col = 1; col <= 3; ++col)
        {
            terrain->pathLayer.set_value(row, col, Colors::Blue);
        }
    }

    // Lab benches (green) - Large work area (4x5)
    for (int row = 1; row <= 4; ++row)
    {
        for (int col = 15; col <= 19; ++col)
        {
            terrain->pathLayer.set_value(row, col, Colors::Green);
        }
    }

    // Security area (yellow) - Bottom-left corner (5x3)
    for (int row = 16; row <= 19; ++row)
    {
        for (int col = 1; col <= 5; ++col)
        {
            terrain->pathLayer.set_value(row, col, Colors::Yellow);
        }
    }

    // Robot/containment area (red) - Bottom-right corner (2x4)
    for (int row = 17; row <= 18; ++row)
    {
        for (int col = 16; col <= 19; ++col)
        {
            terrain->pathLayer.set_value(row, col, Colors::Red);
        }
    }

    // 1. Lead Scientist - Reviews reports at terminal, talks to others
    auto leadScientist = agents->create_behavior_agent("LeadScientist", BehaviorTreeTypes::LeadScientist, Agent::AgentModel::Man);
    leadScientist->set_position(terrain->get_world_position(2, 2));  // Center of blue terminal area (top-left)
    leadScientist->set_color(Vec3(0.2f, 0.2f, 0.8f));  // Blue lab coat
    leadScientist->set_movement_speed(8.0f);
    leadScientist->set_scaling(3.0f);

    // 2. Lab Assistant - Prepares samples, nervous around containment
    auto labAssistant = agents->create_behavior_agent("LabAssistant", BehaviorTreeTypes::LabAssistant, Agent::AgentModel::Man);
    labAssistant->set_position(terrain->get_world_position(2, 17));  // Center of green lab bench area (top-right)
    labAssistant->set_color(Vec3(0.8f, 0.8f, 0.2f));  // Yellow outfit
    labAssistant->set_movement_speed(12.0f);
    labAssistant->set_scaling(3.0f);

    // 3. Security Officer - Patrols, chats with researchers
    auto securityOfficer = agents->create_behavior_agent("SecurityOfficer", BehaviorTreeTypes::SecurityOfficer, Agent::AgentModel::Man);
    securityOfficer->set_position(terrain->get_world_position(17, 3));  // Center of yellow security area (bottom-left)
    securityOfficer->set_color(Vec3(0.1f, 0.1f, 0.1f));  // Dark security uniform
    securityOfficer->set_movement_speed(10.0f);
    securityOfficer->set_scaling(3.0f);

    // 4. Robot (AI Technician + Robot combined) - Erratic behavior, drops items
    auto robot = agents->create_behavior_agent("Robot", BehaviorTreeTypes::RobotTechnician, Agent::AgentModel::Ball);
    robot->set_position(terrain->get_world_position(17, 17));  // Center of red robot work area (bottom-right)
    robot->set_color(Vec3(0.3f, 0.3f, 0.3f));  // Dark metallic gray
    robot->set_movement_speed(6.0f);
    robot->set_scaling(1.f);

    // Position camera to overlook the lab
    auto camera = agents->get_camera_agent();
    camera->set_position(Vec3(-62.0f, 70.0f, terrain->mapSizeInWorld * 0.5f));
    camera->set_pitch(0.610865); // 35 degrees

    // Sound control (these sound functions can be kicked off in a behavior tree node - see the example in L_PlaySound.cpp)
    audioManager->SetVolume(0.3f);
    audioManager->PlaySoundEffect(L"Assets\\Audio\\retro.wav");
    // Uncomment for example on playing music in the engine (must be .wav)
    // audioManager->PlayMusic(L"Assets\\Audio\\motivate.wav");
    // audioManager->PauseMusic(...);
    // audioManager->ResumeMusic(...);
    // audioManager->StopMusic(...);

    // Store important lab positions in blackboards for reference
    leadScientist->get_blackboard().set_value("Terminal Position", terrain->get_world_position(3, 3));    // Blue terminal area
    labAssistant->get_blackboard().set_value("Lab Bench Position", terrain->get_world_position(3, 9));    // Green lab bench area
    labAssistant->get_blackboard().set_value("Containment Position", terrain->get_world_position(9, 9));  // Red containment area
    securityOfficer->get_blackboard().set_value("Patrol Start", terrain->get_world_position(9, 3));       // Yellow security area
    robot->get_blackboard().set_value("Work Area", terrain->get_world_position(9, 9));
}