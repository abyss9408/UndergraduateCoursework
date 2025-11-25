#include <pch.h>
#include "Terrain/TerrainAnalysis.h"
#include "Terrain/MapMath.h"
#include "Agent/AStarAgent.h"
#include "Terrain/MapLayer.h"
#include "Projects/ProjectThree.h"

#include <iostream>

bool ProjectThree::implemented_fog_of_war() const // extra credit
{
    return false;
}

float distance_to_closest_wall(int row, int col)
{
    /*
        Check the euclidean distance from the given cell to every other wall cell,
        with cells outside the map bounds treated as walls, and return the smallest
        distance.  Make use of the is_valid_grid_position and is_wall member
        functions in the global terrain to determine if a cell is within map bounds
        and a wall, respectively.
    */

	float minDistance = std::numeric_limits<float>::max();

	// Get map dimensions
	int mapHeight = terrain->get_map_height();
    int mapWidth = terrain->get_map_width();

	// Iterate through every cell in the map
    for (int r = 0; r < mapHeight; ++r)
    {
        for (int c = 0; c < mapWidth; ++c)
        {
			// Check if the cell is a wall
            if (terrain->is_wall(r, c))
            {
                // Calculate the distance to the wall cell
                float dx = static_cast<float>(col) + 0.5f - (static_cast<float>(c) + 0.5f);
                float dy = static_cast<float>(row) + 0.5f - (static_cast<float>(r) + 0.5f);
                float distance = sqrtf(dx * dx + dy * dy);

                // Update the minimum distance if this one is smaller
                if (distance < minDistance)
                {
                    minDistance = distance;
                }
			}
        }
    }

	// Also check distances to map bounderies (treating them as walls)
	// Top boundary
	float distanceToTop = static_cast<float>(row) + 0.5f;
    if (distanceToTop < minDistance)
    {
        minDistance = distanceToTop;
	}

	// Bottom boundary
    float distanceToBottom = static_cast<float>(mapHeight) - 0.5f - static_cast<float>(row);
    if (distanceToBottom < minDistance)
    {
        minDistance = distanceToBottom;
	}

    // Left boundary
	float distanceToLeft = static_cast<float>(col) + 0.5f;
    if (distanceToLeft < minDistance)
    {
        minDistance = distanceToLeft;
	}

    // Right boundary
    float distanceToRight = static_cast<float>(mapWidth) - 0.5f - static_cast<float>(col);
    if (distanceToRight < minDistance)
    {
        minDistance = distanceToRight;
    }

	// Return the minimum distance found
	return minDistance;
}

bool is_clear_path(int row0, int col0, int row1, int col1)
{
    /*
        Two cells (row0, col0) and (row1, col1) are visible to each other if a line
        between their centerpoints doesn't intersect the four boundary lines of every
        wall cell.  You should puff out the four boundary lines by a very tiny amount
        so that a diagonal line passing by the corner will intersect it.  Make use of the
        line_intersect helper function for the intersection test and the is_wall member
        function in the global terrain to determine if a cell is a wall or not.
    */

    // Calculate centerpoints of both cells
    Vec2 center0(static_cast<float>(col0) + 0.5f, static_cast<float>(row0) + 0.5f);
    Vec2 center1(static_cast<float>(col1) + 0.5f, static_cast<float>(row1) + 0.5f);

    // Early exit: if start and end are the same, path is clear
    if (row0 == row1 && col0 == col1)
    {
        return true;
    }

    // Calculate bounding box of the line to limit wall checking
    int minRow = std::min(row0, row1);
    int maxRow = std::max(row0, row1);
    int minCol = std::min(col0, col1);
    int maxCol = std::max(col0, col1);

    // Expand bounding box by 1 to account for epsilon padding
    minRow = std::max(0, minRow - 1);
    maxRow = std::min(terrain->get_map_height() - 1, maxRow + 1);
    minCol = std::max(0, minCol - 1);
    maxCol = std::min(terrain->get_map_width() - 1, maxCol + 1);

    // Small epsilon value to "puff out" the wall boundaries
    const float epsilon = 0.001f;

    // Only check walls within the bounding box
    for (int r = minRow; r <= maxRow; ++r)
    {
        for (int c = minCol; c <= maxCol; ++c)
        {
            // Only check if this cell is a wall
            if (terrain->is_wall(r, c))
            {
                // Define the four boundary lines of this wall cell (puffed out by epsilon)
                float left = static_cast<float>(c) - epsilon;
                float right = static_cast<float>(c) + 1.0f + epsilon;
                float top = static_cast<float>(r) - epsilon;
                float bottom = static_cast<float>(r) + 1.0f + epsilon;

                // Check intersection with each of the four boundary lines
                // Top edge
                if (line_intersect(center0, center1, Vec2(left, top), Vec2(right, top)))
                {
                    return false;
                }

                // Bottom edge
                if (line_intersect(center0, center1, Vec2(left, bottom), Vec2(right, bottom)))
                {
                    return false;
                }

                // Left edge
                if (line_intersect(center0, center1, Vec2(left, top), Vec2(left, bottom)))
                {
                    return false;
                }

                // Right edge
                if (line_intersect(center0, center1, Vec2(right, top), Vec2(right, bottom)))
                {
                    return false;
                }
            }
        }
    }

    // If we get here, no wall boundaries were intersected
    return true;
}

void analyze_openness(MapLayer<float> &layer)
{
    /*
        Mark every cell in the given layer with the value 1 / (d * d),
        where d is the distance to the closest wall or edge.  Make use of the
        distance_to_closest_wall helper function.  Walls should not be marked.
    */

	// Get map dimensions
    int mapHeight = terrain->get_map_height();
	int mapWidth = terrain->get_map_width();

    // Iterate through every cell in the map
    for (int r = 0; r < mapHeight; ++r)
    {
        for (int c = 0; c < mapWidth; ++c)
        {
            // Check if the cell is a wall
            if (terrain->is_wall(r, c))
            {
				continue; // Skip wall cells
            }

			// Calculate the distance to the closest wall or edge
			float distance = distance_to_closest_wall(r, c);

			// Calculate the openness value
			float opennessValue = 1.0f / (distance * distance);

			// Set the value in the layer
			layer.set_value(r, c, opennessValue);
        }
	}
}

void analyze_visibility(MapLayer<float> &layer)
{
    /*
        Mark every cell in the given layer with the number of cells that
        are visible to it, divided by 160 (a magic number that looks good).  Make sure
        to cap the value at 1.0 as well.

        Two cells are visible to each other if a line between their centerpoints doesn't
        intersect the four boundary lines of every wall cell.  Make use of the is_clear_path
        helper function.
    */

    // Get map dimensions
    int mapHeight = terrain->get_map_height();
    int mapWidth = terrain->get_map_width();

	// Pre-compute list of non-wall cells
	std::vector<std::pair<int, int>> nonWallCells;
	nonWallCells.reserve(mapHeight * mapWidth);

    for (int r = 0; r < mapHeight; ++r)
    {
        for (int c = 0; c < mapWidth; ++c)
        {
            if (!terrain->is_wall(r, c))
            {
                nonWallCells.emplace_back(r, c);
            }
        }
    }

    // Use symmetry: if A can see B, then B can see A
    std::vector<std::vector<int>> visibilityCount(mapHeight, std::vector<int>(mapWidth, 0));

    for (size_t i = 0; i < nonWallCells.size(); ++i)
    {
        int row1 = nonWallCells[i].first;
        int col1 = nonWallCells[i].second;

        // Count self-visibility (a cell can always see itself)
        ++visibilityCount[row1][col1];

        for (size_t j = i + 1; j < nonWallCells.size(); ++j)
        {
            int row2 = nonWallCells[j].first;
            int col2 = nonWallCells[j].second;

            // Check if there's a clear path between the cells
            if (is_clear_path(row1, col1, row2, col2))
            {
                // Due to symmetry, both cells can see each other
                ++visibilityCount[row1][col1];
                ++visibilityCount[row2][col2];
            }
        }

        // Convert counts to final visibility values
        for (const auto& cell : nonWallCells)
        {
            int row = cell.first;
            int col = cell.second;

            // Calculate visibility value: count / 160, capped at 1.0
            float visibilityValue = static_cast<float>(visibilityCount[row][col]) / 160.0f;

            // Cap the value at 1.0
            visibilityValue = std::min(visibilityValue, 1.0f);

            // Set the visibility value in the layer
            layer.set_value(row, col, visibilityValue);
        }
    }
}

void analyze_visible_to_cell(MapLayer<float> &layer, int row, int col)
{
    /*
        For every cell in the given layer mark it with 1.0
        if it is visible to the given cell, 0.5 if it isn't visible but is next to a visible cell,
        or 0.0 otherwise.

        Two cells are visible to each other if a line between their centerpoints doesn't
        intersect the four boundary lines of every wall cell.  Make use of the is_clear_path
        helper function.
    */

    // Get map dimensions
    int mapHeight = terrain->get_map_height();
    int mapWidth = terrain->get_map_width();

    // First pass: mark all cells with visibility to the target cell
    std::vector<std::vector<bool>> isVisible(mapHeight, std::vector<bool>(mapWidth, false));

    for (int r = 0; r < mapHeight; ++r)
    {
        for (int c = 0; c < mapWidth; ++c)
        {
            // Skip wall cells
            if (terrain->is_wall(r, c))
            {
                layer.set_value(r, c, 0.0f);
                continue;
            }

            // Check if this cell has clear line of sight to the target cell
            if (is_clear_path(r, c, row, col))
            {
                layer.set_value(r, c, 1.0f);
                isVisible[r][c] = true;
            }
            else
            {
                // Initially set to 0.0, will be updated in second pass if adjacent to visible cell
                layer.set_value(r, c, 0.0f);
                isVisible[r][c] = false;
            }
        }
    }

    // Second pass: mark cells that are adjacent to visible cells with 0.5 (sniper spots)
    int directions[8][2] = 
    {
        {-1, -1}, {-1, 0}, {-1, 1},  // Top row
        { 0, -1},          { 0, 1},  // Middle row (skip center)
        { 1, -1}, { 1, 0}, { 1, 1}   // Bottom row
    };

    for (int r = 0; r < mapHeight; ++r)
    {
        for (int c = 0; c < mapWidth; ++c)
        {
            // Skip if this cell is a wall or already visible
            if (terrain->is_wall(r, c) || isVisible[r][c])
            {
                continue;
            }

            // Check if any neighbor is visible and reachable
            bool hasReachableVisibleNeighbor = false;

            for (int i = 0; i < 8; ++i)
            {
                int neighborRow = r + directions[i][0];
                int neighborCol = c + directions[i][1];

                // Check if neighbor is within map bounds and NOT a wall
                if (neighborRow >= 0 && neighborRow < mapHeight &&
                    neighborCol >= 0 && neighborCol < mapWidth &&
                    !terrain->is_wall(neighborRow, neighborCol) &&
                    isVisible[neighborRow][neighborCol])
                {
                    // For diagonal moves, check if the path is clear (no corner cutting)
                    bool canReachNeighbor = true;

                    // Check if this is a diagonal move
                    if (directions[i][0] != 0 && directions[i][1] != 0)
                    {
                        // Diagonal move - check the two adjacent cardinal cells
                        int cardinalRow1 = r + directions[i][0];  // Same row as destination
                        int cardinalCol1 = c;                     // Same col as source

                        int cardinalRow2 = r;                     // Same row as source  
                        int cardinalCol2 = c + directions[i][1];  // Same col as destination

                        // Both cardinal cells must be passable for diagonal movement
                        if (terrain->is_wall(cardinalRow1, cardinalCol1) ||
                            terrain->is_wall(cardinalRow2, cardinalCol2))
                        {
                            canReachNeighbor = false;
                        }
                    }

                    if (canReachNeighbor)
                    {
                        hasReachableVisibleNeighbor = true;
                        break;
                    }
                }
            }

            if (hasReachableVisibleNeighbor)
            {
                layer.set_value(r, c, 0.5f);
            }
        }
    }
}

void analyze_agent_vision(MapLayer<float> &layer, const Agent *agent)
{
    /*
        For every cell in the given layer that is visible to the given agent,
        mark it as 1.0, otherwise don't change the cell's current value.

        You must consider the direction the agent is facing.  All of the agent data is
        in three dimensions, but to simplify you should operate in two dimensions, the XZ plane.

        Take the dot product between the view vector and the vector from the agent to the cell,
        both normalized, and compare the cosines directly instead of taking the arccosine to
        avoid introducing floating-point inaccuracy (larger cosine means smaller angle).

        Give the agent a field of view slighter larger than 180 degrees.

        Two cells are visible to each other if a line between their centerpoints doesn't
        intersect the four boundary lines of every wall cell.  Make use of the is_clear_path
        helper function.
    */

	// Get agent position and convert to grid position
	const Vec3 &agentWorldPos = agent->get_position();
	GridPos agentGridPos = terrain->get_grid_position(agentWorldPos);

    // Get agent's forward direction in XZ plane
    Vec3 agentForward3D = agent->get_forward_vector();
    Vec2 agentForward2D(agentForward3D.z, agentForward3D.x);

    // Normalize the forward vector
    agentForward2D.Normalize();

    // Field of view slightly larger than 180 degrees
    const float fovCosineThreshold = -0.087f;

    // Get map dimensions
    int mapHeight = terrain->get_map_height();
    int mapWidth = terrain->get_map_width();

    // Check every cell in the map
    for (int row = 0; row < mapHeight; ++row)
    {
        for (int col = 0; col < mapWidth; ++col)
        {
            // Skip wall cells
            if (terrain->is_wall(row, col))
            {
                continue;
            }

            // Check if there's a clear line of sight
            if (is_clear_path(agentGridPos.row, agentGridPos.col, row, col))
            {
                // Calculate vector from agent to cell (in grid coordinates)
                Vec2 agentToCell
                (
                    static_cast<float>(col) - static_cast<float>(agentGridPos.col),
                    static_cast<float>(row) - static_cast<float>(agentGridPos.row)
                );

                // Skip if this is the agent's own cell (zero vector)
                if (agentToCell.LengthSquared() > 0.001f)
                {
                    // Normalize the vector from agent to cell
                    agentToCell.Normalize();

                    // Calculate dot product (cosine of angle)
                    float cosineAngle = agentForward2D.Dot(agentToCell);

                    // Check if within field of view
                    if (cosineAngle >= fovCosineThreshold)
                    {
                        // Mark cell as visible
                        layer.set_value(row, col, 1.0f);
                    }
                }
                else
                {
                    // Agent's own position is always visible
                    layer.set_value(row, col, 1.0f);
                }
            }
        }
    }
}

void propagate_solo_occupancy(MapLayer<float> &layer, float decay, float growth)
{
    /*
        For every cell in the given layer:

            1) Get the value of each neighbor and apply decay factor
            2) Keep the highest value from step 1
            3) Linearly interpolate from the cell's current value to the value from step 2
               with the growing factor as a coefficient.  Make use of the lerp helper function.
            4) Store the value from step 3 in a temporary layer.
               A float[40][40] will suffice, no need to dynamically allocate or make a new MapLayer.

        After every cell has been processed into the temporary layer, write the temporary layer into
        the given layer;
    */
    
	// Get map dimensions
	int mapHeight = terrain->get_map_height();
	int mapWidth = terrain->get_map_width();

	// Temporary layer to hold new values
	float tempLayer[40][40];

	// Eight directions for neighbor offsets
    int directions[8][2] = 
    {
        {-1, -1}, {-1, 0}, {-1, 1},  // Top row
        { 0, -1},          { 0, 1},  // Middle row
        { 1, -1}, { 1, 0}, { 1, 1}   // Bottom row
    };


	// Iterate through every cell in the map
    for (int r = 0; r < mapHeight; ++r)
    {
        for (int c = 0; c < mapWidth; ++c)
        {
			// Skip wall cells
            if (terrain->is_wall(r, c))
            {
                tempLayer[r][c] = 0.0f;
                continue;
            }

            // Get current cell's value
            float currentValue = layer.get_value(r, c);

            // Find the maximum decayed neighbor influence
            float maxNeighborInfluence = 0.0f;

            // Check all 8 neighbors
            for (int i = 0; i < 8; ++i)
            {
                int neighborRow = r + directions[i][0];
                int neighborCol = c + directions[i][1];

                // Check if neighbor is within map bounds
                if (neighborRow >= 0 && neighborRow < mapHeight &&
                    neighborCol >= 0 && neighborCol < mapWidth &&
                    !terrain->is_wall(neighborRow, neighborCol))
                {
                    // For diagonal neighbors, check if the path is blocked by walls (no corner cutting)
                    bool canReachNeighbor = true;

                    if (directions[i][0] != 0 && directions[i][1] != 0)
                    {
                        // Diagonal move - check the two adjacent cardinal cells
                        int cardinalRow1 = r + directions[i][0];  // Same row as destination
                        int cardinalCol1 = c;                     // Same col as source

                        int cardinalRow2 = r;                     // Same row as source  
                        int cardinalCol2 = c + directions[i][1];  // Same col as destination

                        // Both cardinal cells must be passable for diagonal propagation
                        if (terrain->is_wall(cardinalRow1, cardinalCol1) ||
                            terrain->is_wall(cardinalRow2, cardinalCol2))
                        {
                            canReachNeighbor = false;
                        }
                    }

                    if (canReachNeighbor)
                    {
                        // Get neighbor's influence value
                        float neighborValue = layer.get_value(neighborRow, neighborCol);

                        // Calculate distance to neighbor
                        float distance;
                        if (directions[i][0] != 0 && directions[i][1] != 0)
                        {
                            // Diagonal neighbor: distance = sqrt(2)
                            distance = 1.41421356f;
                        }
                        else
                        {
                            // Cardinal neighbor: distance = 1
                            distance = 1.0f;
                        }

                        // Apply exponential decay: New_Influence = Old_Influence * exp(-1 * Distance * Decay_Factor)
                        float decayedInfluence = neighborValue * expf(-1.0f * distance * decay);

                        // Keep track of maximum decayed influence
                        if (decayedInfluence > maxNeighborInfluence)
                        {
                            maxNeighborInfluence = decayedInfluence;
                        }
                    }
                }
            }

            // Apply linear interpolation between current value and max neighbor influence
            float newValue = lerp(currentValue, maxNeighborInfluence, growth);

            // Store result in temporary layer
            tempLayer[r][c] = newValue;
        }
    }

    // Copy values from temporary layer back to the main layer
    for (int r = 0; r < mapHeight; ++r)
    {
        for (int c = 0; c < mapWidth; ++c)
        {
            layer.set_value(r, c, tempLayer[r][c]);
        }
    }
}

void normalize_solo_occupancy(MapLayer<float> &layer)
{
    /*
        Determine the maximum value in the given layer, and then divide the value
        for every cell in the layer by that amount.  This will keep the values in the
        range of [0, 1].  Negative values should be left unmodified.
    */

    // Get map dimensions
    int mapHeight = terrain->get_map_height();
    int mapWidth = terrain->get_map_width();

    // First pass: find the maximum positive value in the layer
    float maxValue = 0.0f;

    for (int row = 0; row < mapHeight; ++row)
    {
        for (int col = 0; col < mapWidth; ++col)
        {
            // Skip wall cells
            if (terrain->is_wall(row, col))
            {
                continue;
            }

            float cellValue = layer.get_value(row, col);

            // Only consider positive values for maximum calculation
            if (cellValue > maxValue)
            {
                maxValue = cellValue;
            }
        }
    }

    // If maximum value is 0 or negative, no normalization needed
    if (maxValue <= 0.0f)
    {
        return;
    }

    // Second pass: normalize all values by dividing by the maximum
    for (int row = 0; row < mapHeight; ++row)
    {
        for (int col = 0; col < mapWidth; ++col)
        {
            if (terrain->is_wall(row, col))
            {
                continue;
            }

            float cellValue = layer.get_value(row, col);

            // Only normalize non-negative values
            if (cellValue >= 0.0f)
            {
                float normalizedValue = cellValue / maxValue;
                layer.set_value(row, col, normalizedValue);
            }
        }
    }
}

void enemy_field_of_view(MapLayer<float> &layer, float fovAngle, float closeDistance, float occupancyValue, AStarAgent *enemy)
{
    /*
        First, clear out the old values in the map layer by setting any negative value to 0.
        Then, for every cell in the layer that is within the field of view cone, from the
        enemy agent, mark it with the occupancy value.  Take the dot product between the view
        vector and the vector from the agent to the cell, both normalized, and compare the
        cosines directly instead of taking the arccosine to avoid introducing floating-point
        inaccuracy (larger cosine means smaller angle).

        If the tile is close enough to the enemy (less than closeDistance),
        you only check if it's visible to enemy.  Make use of the is_clear_path
        helper function.  Otherwise, you must consider the direction the enemy is facing too.
        This creates a radius around the enemy that the player can be detected within, as well
        as a fov cone.
    */

	// Get map dimensions
	int mapHeight = terrain->get_map_height();
	int mapWidth = terrain->get_map_width();

	// First pass: clear old values in the layer
    for (int row = 0; row < mapHeight; ++row)
    {
        for (int col = 0; col < mapWidth; ++col)
        {
            float cellValue = layer.get_value(row, col);
            if (cellValue < 0.0f)
            {
                layer.set_value(row, col, 0.0f);
            }
        }
    }

    // Get enemy's world position and convert to grid position
    const Vec3 &enemyWorldPos = enemy->get_position();
    const GridPos enemyGridPos = terrain->get_grid_position(enemyWorldPos);

    // Verify enemy is on valid terrain
    if (!terrain->is_valid_grid_position(enemyGridPos))
    {
        return;
    }

    // Get enemy's forward vector in the XZ plane (ignore Y component)
    const Vec3 enemyForward3D = enemy->get_forward_vector();
    Vec2 enemyForward2D(enemyForward3D.z, enemyForward3D.x);

	// Normalize the forward vector
	enemyForward2D.Normalize();

    // Convert FOV angle from degrees to cosine threshold
    float fovRadians = fovAngle * (3.14159265f / 180.0f);  // Convert to radians
    float fovCosineThreshold = cosf(fovRadians * 0.5f);     // Half angle for cone

    // Check every cell in the map
    for (int row = 0; row < mapHeight; ++row)
    {
        for (int col = 0; col < mapWidth; ++col)
        {
            // Skip wall cells
            if (terrain->is_wall(row, col))
            {
                continue;
            }

            // Skip the enemy's own cell
            if (row == enemyGridPos.row && col == enemyGridPos.col)
            {
                continue;
            }

            // Calculate distance from enemy to cell
            float dx = static_cast<float>(col) + 0.5f - (static_cast<float>(enemyGridPos.col) + 0.5f);
            float dy = static_cast<float>(row) + 0.5f - (static_cast<float>(enemyGridPos.row) + 0.5f);
            float distanceToCell = sqrtf(dx * dx + dy * dy);

            // Check if there's clear line of sight first (required for both close and FOV detection)
            if (!is_clear_path(enemyGridPos.row, enemyGridPos.col, row, col))
            {
                continue;  // No line of sight, skip this cell
            }

            bool canDetect = false;

            // Check if cell is within close detection radius
            if (distanceToCell <= closeDistance)
            {
                // Close enough - no need to check FOV angle
                canDetect = true;
            }
            else
            {
                // Not in close range - check if it's within the FOV cone
                if (distanceToCell > 0.0f)
                {
                    // Calculate normalized vector from enemy to cell
                    Vec2 enemyToCell(dx / distanceToCell, dy / distanceToCell);

                    // Calculate dot product to determine if cell is within field of view
                    float dotProduct = enemyForward2D.Dot(enemyToCell);

                    // Check if cell is within FOV cone
                    if (dotProduct >= fovCosineThreshold)
                    {
                        canDetect = true;
                    }
                }
            }

            if (canDetect)
            {
                // Mark with negative occupancy value to indicate enemy detection zone
                layer.set_value(row, col, occupancyValue);
            }
        }
    }
}

bool enemy_find_player(MapLayer<float> &layer, AStarAgent *enemy, Agent *player)
{
    /*
        Check if the player's current tile has a negative value, ie in the fov cone
        or within a detection radius.
    */

    const auto &playerWorldPos = player->get_position();
    const auto playerGridPos = terrain->get_grid_position(playerWorldPos);

    // verify a valid position was returned
    if (terrain->is_valid_grid_position(playerGridPos))
    {
        if (layer.get_value(playerGridPos) < 0.0f)
        {
            return true;
        }
    }

    // player isn't in the detection radius or fov cone, OR somehow off the map
    return false;
}

bool enemy_seek_player(MapLayer<float> &layer, AStarAgent *enemy)
{
    /*
        Attempt to find a cell with the highest nonzero value (normalization may
        not produce exactly 1.0 due to floating point error), and then set it as
        the new target, using enemy->path_to.

        If there are multiple cells with the same highest value, then pick the
        cell closest to the enemy.

        Return whether a target cell was found.
    */

    // Get map directions
    int mapHeight = terrain->get_map_height();
    int mapWidth = terrain->get_map_width();

    // Get enemy's current grid position for distance calculations
    const Vec3 &enemyWorldPos = enemy->get_position();
    const GridPos enemyGridPos = terrain->get_grid_position(enemyWorldPos);

    // Variables to track the best target
    float highestValue = 0.0f;
    GridPos bestTarget = { -1, -1 };
    float closestDistance = std::numeric_limits<float>::max();

    // Search all cells for the highest positive occupancy value
    for (int row = 0; row < mapHeight; ++row)
    {
        for (int col = 0; col < mapWidth; ++col)
        {
            float cellValue = layer.get_value(row, col);

            // Only consider positive values (ignore negative FOV values and zeros)
            if (cellValue > 0.0f)
            {
                // Check if this is a higher value than we've found
                if (cellValue > highestValue)
                {
                    // New highest value found
                    highestValue = cellValue;
                    bestTarget.row = row;
                    bestTarget.col = col;

                    // Calculate distance for this new best candidate
                    float dx = static_cast<float>(col - enemyGridPos.col);
                    float dy = static_cast<float>(row - enemyGridPos.row);
                    closestDistance = sqrtf(dx * dx + dy * dy);
                }
                else if (cellValue == highestValue)
                {
                    // Same value as current best - check if it's closer
                    float dx = static_cast<float>(col - enemyGridPos.col);
                    float dy = static_cast<float>(row - enemyGridPos.row);
                    float distance = sqrtf(dx * dx + dy * dy);

                    if (distance < closestDistance)
                    {
                        // This cell is closer, make it the new target
                        bestTarget.row = row;
                        bestTarget.col = col;
                        closestDistance = distance;
                    }
                }
            }
        }
    }

    // Check if we found a valid target
    if (bestTarget.row != -1 && bestTarget.col != -1)
    {
        // Convert grid position to world position and set as target
        const Vec3 targetWorldPos = terrain->get_world_position(bestTarget);
        enemy->path_to(targetWorldPos);
        return true;  // Target found and set
    }

    return false;  // No suitable target found
}
