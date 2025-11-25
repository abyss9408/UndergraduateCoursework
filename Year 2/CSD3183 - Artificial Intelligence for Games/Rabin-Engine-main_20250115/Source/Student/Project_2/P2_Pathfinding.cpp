#include <pch.h>
#include "Projects/ProjectTwo.h"
#include "P2_Pathfinding.h"

#pragma region Extra Credit 
bool ProjectTwo::implemented_floyd_warshall()
{
    return false;
}

bool ProjectTwo::implemented_goal_bounding()
{
    return false;
}
#pragma endregion

AStarPather::AStarPather() : currentNode(nullptr)
{
}

bool AStarPather::initialize()
{
    // handle any one-time setup requirements you have

    /*
        If you want to do any map-preprocessing, you'll need to listen
        for the map change message.  It'll look something like this:

        Callback cb = std::bind(&AStarPather::your_function_name, this);
        Messenger::listen_for_message(Messages::MAP_CHANGE, cb);

        There are other alternatives to using std::bind, so feel free to mix it up.
        Callback is just a typedef for std::function<void(void)>, so any std::invoke'able
        object that std::function can wrap will suffice.
    */

    // Pre-allocate the space space for maximum map size (your existing approach)
	openList.reserve(Terrain::maxMapHeight * Terrain::maxMapWidth);
    for (int i = 0; i < Terrain::maxMapHeight; ++i)
    {
        for (int j = 0; j < Terrain::maxMapWidth; ++j)
        {
            nodes[i][j] = Node(i, j);
        }
    }

    // Listen for map changes to precompute neighbors when terrain is available
    Callback cb = std::bind(&AStarPather::onMapChange, this);
    Messenger::listen_for_message(Messages::MAP_CHANGE, cb);

    return true;
}

void AStarPather::shutdown()
{
    /*
        Free any dynamically allocated memory or any other general house-
        keeping you need to do during shutdown.
    */

    // Clear any remaining data
    clearNodes();
}

PathResult AStarPather::compute_path(PathRequest& request)
{
    /*
        This is where you handle pathing requests, each request has several fields:

        start/goal - start and goal world positions
        path - where you will build the path upon completion, path should be
            start to goal, not goal to start
        heuristic - which heuristic calculation to use
        weight - the heuristic weight to be applied
        newRequest - whether this is the first request for this path, should generally
            be true, unless single step is on

        smoothing - whether to apply smoothing to the path
        rubberBanding - whether to apply rubber banding
        singleStep - whether to perform only a single A* step
        debugColoring - whether to color the grid based on the A* state:
            closed list nodes - yellow
            open list nodes - blue

            use terrain->set_color(row, col, Colors::YourColor);
            also it can be helpful to temporarily use other colors for specific states
            when you are testing your algorithms

        method - which algorithm to use: A*, Floyd-Warshall, JPS+, or goal bounding,
            will be A* generally, unless you implement extra credit features

        The return values are:
            PROCESSING - a path hasn't been found yet, should only be returned in
                single step mode until a path is found
            COMPLETE - a path to the goal was found and has been built in request.path
            IMPOSSIBLE - a path from start to goal does not exist, do not add start position to path
    */

    if (request.newRequest)
    {
        initializeSearch(request);
    }

    if (request.settings.singleStep)
    {
        return performSingleStep(request);
    }
    else
    {
        return performFullSearch(request);
    }
}

void AStarPather::initializeSearch(const PathRequest& request)
{
    clearNodes();

    // Convert world positions to grid positions
    startPos = terrain->get_grid_position(request.start);
    goalPos = terrain->get_grid_position(request.goal);

    // Validate start and goal positions
    if (!terrain->is_valid_grid_position(startPos) || !terrain->is_valid_grid_position(goalPos) ||
        terrain->is_wall(startPos) || terrain->is_wall(goalPos))
    {
        return; // Invalid positions, no path can be computed
    }

    // Initialize start node
    Node& startNode = nodes[startPos.row][startPos.col];

    startNode.givenCost = 0.0f;
    startNode.finalCost = calculateHeuristic(startPos, goalPos, request.settings.heuristic) * request.settings.weight;
    startNode.state = Node::State::OPEN;

    openList.push(&startNode);
    currentNode = nullptr;
}

float AStarPather::calculateHeuristic(const GridPos& from, const GridPos& to, Heuristic heuristicType) const
{
    float dx = static_cast<float>(std::abs(to.row - from.row));
    float dy = static_cast<float>(std::abs(to.col - from.col));

    switch (heuristicType)
    {
    case Heuristic::OCTILE:
    {
        float smaller = std::min(dx, dy);
        float larger = std::max(dx, dy);
        return smaller * DIAGONAL_COST + (larger - smaller);
    }

    case Heuristic::MANHATTAN:
        return dx + dy;

    case Heuristic::CHEBYSHEV:
        return std::max(dx, dy);

    case Heuristic::EUCLIDEAN:
        return std::sqrt(dx * dx + dy * dy);

    case Heuristic::INCONSISTENT:
        if ((from.row + from.col) % 2 > 0)
        {
            return std::sqrt(dx * dx + dy * dy);
        }
        return 0.0f;
    };
}

void AStarPather::precomputeNeighbors()
{
    const int height = terrain->get_map_height();
    const int width = terrain->get_map_width();

    for (int row = 0; row < height; ++row)
    {
        for (int col = 0; col < width; ++col)
        {
            Node& node = nodes[row][col];
            node.validNeighbors = 0; // Clear all bits

            // Skip if current position is a wall
            if (terrain->is_wall(row, col))
                continue;

            // Use lookup table for direction mappings
            static constexpr int DIAG_SIDE_MAP[4][2] = 
            {
                {0, 2}, // NE: N, E
                {2, 4}, // SE: E, S  
                {4, 6}, // SW: S, W
                {6, 0}  // NW: W, N
            };

            // Check cardinals first
            for (int dir = 0; dir < 8; dir += 2)
            {
                int newRow = row + DIRECTION_OFFSETS[dir][0];
                int newCol = col + DIRECTION_OFFSETS[dir][1];

                if (newRow >= 0 && newRow < height && newCol >= 0 && newCol < width &&
                    !terrain->is_wall(newRow, newCol))
                {
                    node.validNeighbors |= (1 << dir);
                }
            }

            // Check diagonals with corner cutting prevention
            for (int i = 0; i < 4; ++i)
            {
                int diagDir = 1 + i * 2; // 1,3,5,7
                int side1Dir = DIAG_SIDE_MAP[i][0];
                int side2Dir = DIAG_SIDE_MAP[i][1];

                if ((node.validNeighbors & (1 << side1Dir)) &&
                    (node.validNeighbors & (1 << side2Dir)))
                {

                    int newRow = row + DIRECTION_OFFSETS[diagDir][0];
                    int newCol = col + DIRECTION_OFFSETS[diagDir][1];

                    if (newRow >= 0 && newRow < height && newCol >= 0 && newCol < width &&
                        !terrain->is_wall(newRow, newCol))
                    {
                        node.validNeighbors |= (1 << diagDir);
                    }
                }
            }
        }
    }
}

void AStarPather::expandNeighbors(Node* currentNode, PathRequest& request)
{
    uint8_t neighbors = currentNode->validNeighbors;

    // Early exit if no valid neighbors
    if (neighbors == 0) return;

    const float currentGCost = currentNode->givenCost;
    const float weight = request.settings.weight;
    const Heuristic heuristicType = request.settings.heuristic;

    // Loop through each bit (direction)
    for (int dir = 0; dir < 8; ++dir)
    {
        // Check if this direction bit is set
        if (!(neighbors & (1 << dir))) continue;

        // Calculate neighbor position using precomputed offsets
        int neighborRow = currentNode->gridPos.row + DIRECTION_OFFSETS[dir][0];
        int neighborCol = currentNode->gridPos.col + DIRECTION_OFFSETS[dir][1];

        Node& childNode = nodes[neighborRow][neighborCol];

        // Calculate movement cost (diagonal vs cardinal)
        float movementCost = (dir & 1) ? DIAGONAL_COST : 1.0f;
        float newGCost = currentGCost + movementCost;

        // Early exit if this path is definitely worse
        if (childNode.state != Node::State::NONE && newGCost >= childNode.givenCost)
            continue;

        // Compute costs
        float newHCost = calculateHeuristic({ neighborRow, neighborCol }, goalPos,
            request.settings.heuristic) * request.settings.weight;
        float newFCost = newGCost + newHCost;

        // Process child node (same logic as before)
        if (childNode.state == Node::State::NONE)
        {
            childNode.parent = currentNode;
            childNode.givenCost = newGCost;
            childNode.finalCost = newFCost;
            childNode.state = Node::State::OPEN;
            openList.push(&childNode);
        }
        else if ((childNode.state == Node::State::OPEN || childNode.state == Node::State::CLOSED) &&
            newFCost < childNode.finalCost)
        {
            if (childNode.state == Node::State::OPEN)
            {
                openList.remove(&childNode);
            }

            childNode.parent = currentNode;
            childNode.givenCost = newGCost;
            childNode.finalCost = newFCost;
            childNode.state = Node::State::OPEN;
            openList.push(&childNode);
        }
    }
}

PathResult AStarPather::performSingleStep(PathRequest& request)
{
    if (openList.empty())
    {
        return PathResult::IMPOSSIBLE;
    }

    // Find and get the node with lowest final cost
    currentNode = openList.pop();

    // Check if we reached the goal
    if (currentNode->gridPos == goalPos)
    {
        reconstructPath(currentNode, request);
        applySmoothingAndRubberBanding(request);

        if (request.settings.debugColoring)
        {
            updateDebugColors();
        }

        return PathResult::COMPLETE;
    }

    // Place parentNode on the Closed List
    currentNode->state = Node::State::CLOSED;

    // Expand neighbors
    expandNeighbors(currentNode, request);

    if (request.settings.debugColoring)
    {
        updateDebugColors();
    }

    return PathResult::PROCESSING;
}

PathResult AStarPather::performFullSearch(PathRequest& request)
{
    while (!openList.empty())
    {
        PathResult result = performSingleStep(request);
        if (result != PathResult::PROCESSING)
        {
            return result;
        }
    }

    return PathResult::IMPOSSIBLE;
}

void AStarPather::reconstructPath(Node* goalNode, PathRequest& request)
{
    request.path.clear();
    Node* current = goalNode;

    // Build path from goal to start, inserting at front
    while (current != nullptr)
    {
        Vec3 worldPos = terrain->get_world_position(current->gridPos);
        request.path.emplace_front(worldPos);  // Insert at beginning
        current = current->parent;
    }
}

void AStarPather::updateDebugColors() const
{
    int mapHeight = terrain->get_map_height();
    int mapWidth = terrain->get_map_width();

    for (int row = 0; row < mapHeight; ++row)
    {
        for (int col = 0; col < mapWidth; ++col)
        {
            const Node& node = nodes[row][col];

            if (node.state == Node::State::OPEN)
            {
                terrain->set_color(row, col, Colors::Blue);
            }
            if (node.state == Node::State::CLOSED)
            {
                terrain->set_color(row, col, Colors::Yellow);
            }
        }
    }
}

void AStarPather::applySmoothingAndRubberBanding(PathRequest& request) const
{
    if (request.settings.rubberBanding)
    {
        rubberBandPath(request.path);
    }

    if (request.settings.smoothing)
    {
        smoothPath(request.path);
    }
}

void AStarPather::smoothPath(WaypointList& path) const
{
    if (path.size() < 2) return;

    // First, check if we need to add points back due to rubber banding
    // Requirement: When combined with rubberbanding, must add back points if < 1.5 grid units between waypoints
    WaypointList densifiedPath;
    densifiedPath.emplace_back(path.front());

    auto current = path.begin();
    auto next = std::next(current);

    while (next != path.end())
    {
        Vec3 currentPos = *current;
        Vec3 nextPos = *next;

        // Calculate distance in grid units
        float distance = Vec3::Distance(currentPos, nextPos);
        float gridDistance = distance / globalScalar; // Convert to grid units

        // If distance is >= 1.5 grid units, add intermediate points
        if (gridDistance >= 1.5f)
        {
            int numSegments = static_cast<int>(std::ceil(gridDistance / 1.0f)); // Add points every ~1 grid unit

            for (int i = 1; i < numSegments; ++i)
            {
                float t = static_cast<float>(i) / static_cast<float>(numSegments);
                Vec3 interpolated = Vec3::Lerp(currentPos, nextPos, t);
                densifiedPath.emplace_back(interpolated);
            }
        }

        densifiedPath.emplace_back(nextPos);
        current = next;
        ++next;
    }

    // Now apply Catmull-Rom spline smoothing
    // Requirement: Must be 3 spline points in-between every waypoint
    if (densifiedPath.size() < 2)
    {
        path = densifiedPath;
        return;
    }

    WaypointList smoothedPath;
    smoothedPath.emplace_back(densifiedPath.front()); // Keep start point

    auto it = densifiedPath.begin();
    //++it; // Skip first point

    while (it != densifiedPath.end() && std::next(it) != densifiedPath.end())
    {
        Vec3 p0 = (it == densifiedPath.begin()) ? *it : *std::prev(it);
        Vec3 p1 = *it;
        Vec3 p2 = *std::next(it);
        Vec3 p3 = (std::next(it, 2) == densifiedPath.end()) ? p2 : *std::next(it, 2);

        smoothedPath.emplace_back(p1); // Keep original waypoint

        // Add exactly 3 interpolated points between waypoints (requirement)
        for (int i = 1; i <= 3; ++i)
        {
            float t = i / 4.0f; // t = 0.25, 0.5, 0.75
            float t2 = t * t;
            float t3 = t2 * t;

            // Catmull-Rom spline formula
            Vec3 interpolated = 0.5f * 
            (
                (2.0f * p1) +
                (-p0 + p2) * t +
                (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3) * t2 +
                (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3
            );

            smoothedPath.emplace_back(interpolated);
        }
        ++it;
    }

    smoothedPath.emplace_back(densifiedPath.back()); // Keep end point
    path = smoothedPath;
}

void AStarPather::rubberBandPath(WaypointList& path) const
{
    if (path.size() < 3) return;

    bool changed = true;
    while (changed)
    {
        changed = false;
        auto current = path.begin();

        while (current != path.end())
        {
            auto middle = std::next(current);
            if (middle == path.end()) break;

            auto next = std::next(middle);
            if (next == path.end()) break;

            // Get the three consecutive points
            Vec3 point1 = *current;
            Vec3 point2 = *middle;  // Middle point (candidate for removal)
            Vec3 point3 = *next;

            // Convert to grid positions
            GridPos grid1 = terrain->get_grid_position(point1);
            GridPos grid2 = terrain->get_grid_position(point2);
            GridPos grid3 = terrain->get_grid_position(point3);

            // Check if we can eliminate the middle point
            if (canEliminateMiddleNode(grid1, grid2, grid3))
            {
                // Remove the middle point from the list
                path.erase(middle);
                changed = true;
                break; // Start over from the beginning to be safe
            }
            else
            {
                // Can't eliminate middle point, move to next set of three
                ++current;
            }
        }
    }
}

bool AStarPather::canEliminateMiddleNode(const GridPos& node1, const GridPos& node2, const GridPos& node3) const
{
    // First, do a direct line-of-sight check using the same algorithm as A*
    if (!checkDiagonalMovementValid(node1, node3))
    {
        return false;
    }

    // Then do the rectangle check as additional safety
    int minRow = std::min({ node1.row, node2.row, node3.row });
    int maxRow = std::max({ node1.row, node2.row, node3.row });
    int minCol = std::min({ node1.col, node2.col, node3.col });
    int maxCol = std::max({ node1.col, node2.col, node3.col });

    // Check EVERY tile in the rectangle
    for (int row = minRow; row <= maxRow; ++row)
    {
        for (int col = minCol; col <= maxCol; ++col)
        {
            GridPos checkPos = { row, col };

            // If ANY tile in the rectangle is invalid or a wall, 
            // we CANNOT eliminate the middle node
            if (!terrain->is_valid_grid_position(checkPos) || terrain->is_wall(checkPos))
            {
                return false;
            }
        }
    }

    return true;
}

bool AStarPather::checkDiagonalMovementValid(const GridPos& start, const GridPos& end) const
{
    // Use the exact same logic as your A* pathfinding for movement validation
    GridPos current = start;

    while (current.row != end.row || current.col != end.col)
    {
        // Determine the next step towards the goal
        GridPos next = current;

        int deltaRow = end.row - current.row;
        int deltaCol = end.col - current.col;

        // Move one step at a time using the same rules as A*
        if (deltaRow != 0 && deltaCol != 0)
        {
            // Diagonal movement - use the same validation as getNeighbors()
            int stepRow = (deltaRow > 0) ? 1 : -1;
            int stepCol = (deltaCol > 0) ? 1 : -1;

            next.row += stepRow;
            next.col += stepCol;

            // Check diagonal movement validity (same as in getNeighbors)
            GridPos side1 = { current.row + stepRow, current.col };
            GridPos side2 = { current.row, current.col + stepCol };

            if (!terrain->is_valid_grid_position(side1) || terrain->is_wall(side1) ||
                !terrain->is_valid_grid_position(side2) || terrain->is_wall(side2) ||
                !terrain->is_valid_grid_position(next) || terrain->is_wall(next))
            {
                return false;
            }
        }
        else if (deltaRow != 0)
        {
            // Vertical movement
            next.row += (deltaRow > 0) ? 1 : -1;

            if (!terrain->is_valid_grid_position(next) || terrain->is_wall(next))
            {
                return false;
            }
        }
        else if (deltaCol != 0)
        {
            // Horizontal movement
            next.col += (deltaCol > 0) ? 1 : -1;

            if (!terrain->is_valid_grid_position(next) || terrain->is_wall(next))
            {
                return false;
            }
        }

        current = next;
    }

    return true;
}

void AStarPather::onMapChange()
{
    // Now we can safely access terrain and precompute neighbors
    precomputeNeighbors();
}

void AStarPather::clearNodes()
{
    // Reset all nodes to initial state
    int mapHeight = terrain->get_map_height();
    int mapWidth = terrain->get_map_width();

    for (int row = 0; row < mapHeight; ++row)
    {
        for (int col = 0; col < mapWidth; ++col)
        {
            Node& node = nodes[row][col];
            node.parent = nullptr;
            node.state = Node::State::NONE;
        }
    }

    // Clear the open list
    while (!openList.empty())
    {
        openList.pop();
    }
}

void CustomPriorityQueue::heapifyUp(size_t index)
{
    while (index > 0)
    {
        size_t parentIndex = parent(index);

        // Min-heap: parent should have lower or equal finalCost
        if (heap[parentIndex]->finalCost <= heap[index]->finalCost)
            break;

        std::swap(heap[parentIndex], heap[index]);
        index = parentIndex;
    }
}

void CustomPriorityQueue::heapifyDown(size_t index)
{
    size_t heapSize = heap.size();

    while (true)
    {
        size_t smallest = index;
        size_t left = leftChild(index);
        size_t right = rightChild(index);

        // Find the smallest among parent, left child, and right child
        if (left < heapSize && heap[left]->finalCost < heap[smallest]->finalCost)
            smallest = left;

        if (right < heapSize && heap[right]->finalCost < heap[smallest]->finalCost)
            smallest = right;

        // If heap property is satisfied, we're done
        if (smallest == index)
            break;

        std::swap(heap[index], heap[smallest]);
        index = smallest;
    }
}

void CustomPriorityQueue::push(Node* node)
{
    heap.emplace_back(node);
    heapifyUp(heap.size() - 1);
}

Node* CustomPriorityQueue::pop()
{
    if (heap.empty())
        return nullptr;

    Node* result = heap[0];

    // Move last element to root and heapify down
    heap[0] = heap.back();
    heap.pop_back();

    if (!heap.empty())
        heapifyDown(0);

    return result;
}

void CustomPriorityQueue::remove(Node* nodeToRemove)
{
    // Find the node in the heap
    for (size_t i = 0; i < heap.size(); ++i)
    {
        if (heap[i] == nodeToRemove)
        {
            // Replace with last element
            heap[i] = heap.back();
            heap.pop_back();

            // If we removed the last element, we're done
            if (i >= heap.size())
                return;

            // Restore heap property by trying both up and down
            // Compare with parent first
            if (i > 0 && heap[i]->finalCost < heap[parent(i)]->finalCost)
            {
                heapifyUp(i);
            }
            else
            {
                heapifyDown(i);
            }
            return;
        }
    }
}