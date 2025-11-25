#pragma once
#include "Misc/PathfindingDetails.hpp"
#include <vector>
#include <queue>

// Node structure to represent a point in the grid
struct Node
{
    Node* parent; // Pointer to the parent node
    GridPos gridPos; // Position in the grid
    float finalCost; // Final cost f(x) = g(x) + h(x)
    float givenCost; // Cost from start to this node
    enum class State : uint8_t { NONE, OPEN, CLOSED } state; // Node state for open/closed lists
    uint8_t validNeighbors; // Bit flags for 8 directions: N, NE, E, SE, S, SW, W, NW

    Node() : parent(nullptr), gridPos{ -1, -1 }, finalCost(0.0f), givenCost(0.0f),
        state(State::NONE), validNeighbors(0)
    {
    }

    Node(int row, int col) : parent(nullptr), gridPos{ row, col }, finalCost(0.0f),
        givenCost(0.0f), state(State::NONE), validNeighbors(0)
    {
    }
};

class CustomPriorityQueue
{
private:
    std::vector<Node*> heap;

    // Helper functions for heap operations
    size_t parent(size_t i) const { return (i - 1) / 2; }
    size_t leftChild(size_t i) const { return 2 * i + 1; }
    size_t rightChild(size_t i) const { return 2 * i + 2; }

    void heapifyUp(size_t index);
    void heapifyDown(size_t index);

public:
    bool empty() const { return heap.empty(); }
    size_t size() const { return heap.size(); }
    void clear() { heap.clear(); }

    void push(Node* node);
    Node* pop(); // Returns and removes the lowest cost node
    void remove(Node* node); // Remove specific node from heap
	void reserve(size_t capacity) { heap.reserve(capacity); }

    // For debugging
    Node* top() const { return heap.empty() ? nullptr : heap[0]; }
};

class AStarPather
{
public:
    /*
        The class should be default constructible, so you might need to define a constructor.
        If needed, you can modify the framework where the class is constructed in the
        initialize functions of ProjectTwo and ProjectThree.
    */
    AStarPather();

    /* ************************************************** */
    // DO NOT MODIFY THESE SIGNATURES
    bool initialize();
    void shutdown();
    PathResult compute_path(PathRequest& request);
    /* ************************************************** */

    /*
        You should create whatever functions, variables, or classes you need.
        It doesn't all need to be in this header and cpp, structure it whatever way
        makes sense to you.
    */
private:
    // Search space - 2D array of nodes
    Node nodes[Terrain::maxMapHeight][Terrain::maxMapWidth];

    // Open list for A* search, using a custom priority queue
    CustomPriorityQueue openList;

    Node* currentNode;
    GridPos startPos, goalPos;

    // Direction offsets for 8-directional movement (N, NE, E, SE, S, SW, W, NW)
    static constexpr int DIRECTION_OFFSETS[8][2] = 
    {
        {-1,  0}, // N  (North)
        {-1,  1}, // NE (Northeast) 
        { 0,  1}, // E  (East)
        { 1,  1}, // SE (Southeast)
        { 1,  0}, // S  (South)
        { 1, -1}, // SW (Southwest)
        { 0, -1}, // W  (West)
        {-1, -1}  // NW (Northwest)
    };

    // Direction bit masks
    static constexpr uint8_t DIR_N = 1 << 0;  // 0b00000001
    static constexpr uint8_t DIR_NE = 1 << 1;  // 0b00000010
    static constexpr uint8_t DIR_E = 1 << 2;  // 0b00000100
    static constexpr uint8_t DIR_SE = 1 << 3;  // 0b00001000
    static constexpr uint8_t DIR_S = 1 << 4;  // 0b00010000
    static constexpr uint8_t DIR_SW = 1 << 5;  // 0b00100000
    static constexpr uint8_t DIR_W = 1 << 6;  // 0b01000000
    static constexpr uint8_t DIR_NW = 1 << 7;  // 0b10000000

    static constexpr float DIAGONAL_COST = 1.41421356f; // sqrt(2) for diagonal movement

    // Helper functions
    void clearNodes();
    void initializeSearch(const PathRequest& request);
    float calculateHeuristic(const GridPos& from, const GridPos& to, Heuristic heuristicType) const;
    void precomputeNeighbors();
    void onMapChange();          // Message callback for MAP_CHANGE
    void expandNeighbors(Node* currentNode, PathRequest& request); // Use in search loop

    void reconstructPath(Node* goalNode, PathRequest& request);
    void updateDebugColors() const;
    PathResult performSingleStep(PathRequest& request);
    PathResult performFullSearch(PathRequest& request);

    // Post-processing
    void applySmoothingAndRubberBanding(PathRequest& request) const;
    void smoothPath(WaypointList& path) const;
    void rubberBandPath(WaypointList& path) const;
    bool canEliminateMiddleNode(const GridPos& node1, const GridPos& node2, const GridPos& node3) const;
    bool checkDiagonalMovementValid(const GridPos& start, const GridPos& end) const;
};