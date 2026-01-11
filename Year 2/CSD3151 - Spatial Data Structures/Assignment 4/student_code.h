#ifndef _STUDENTCODE
#define _STUDENTCODE

#define TEST_GEOMLIB 0
#define SCAN_LINE    1
#define RAY_TRACE    2   

#ifdef DEFMAIN
#define MAIN DEFMAIN
#else

// @@ Choose which one of three main programs to run.  Legal choices are:
// #define MAIN	TEST_GEOMLIB
// #define MAIN	SCAN_LINE
// #define MAIN	RAY_TRACE
#define MAIN	SCAN_LINE

#endif

const int thousand = 1000;
const int million = 1000000;

// @@ TRIANGLE COUNT: Set GOAL to the (approximate) number of
// triangles you'd like for this run.
#if MAIN==RAY_TRACE
static int GOAL = 100;
#else
static int GOAL = 10 * thousand;
#endif

static int polyCount = 1;      // Produces 4*polyCount^2 triangles per ellipsoid
static int ellipsoidCount = 0; // Produces (2*ellipsoidCount+1)^2 ellipsoids
static int grndCount = 4;      // Produces 2*groundCount^2 triangles.

#include "geomlib.h"

// Forward declaration
class Object;

// Enhanced Triangle structure to store object reference for ray tracing
struct TriangleData
{
    Triangle triangle;
    Object* obj;           // Object reference for material properties
    vec3 normals[3];      // Vertex normals for smooth shading (optional)
    bool hasVertexNormals; // Flag to indicate if vertex normals are available

    TriangleData() : obj(nullptr), hasVertexNormals(false) {}
    TriangleData(const Triangle& tri, Object* object) : triangle(tri), obj(object), hasVertexNormals(false) {}
    TriangleData(const Triangle& tri, Object* object, const vec3& n0, const vec3& n1, const vec3& n2) 
        : triangle(tri), obj(object), hasVertexNormals(true) 
    {
        normals[0] = n0;
        normals[1] = n1;
        normals[2] = n2;
    }
};

// Event structure for the sweep algorithm
struct SweepEvent
{
    float position;
    enum Type { START, END, PLANAR } type;
    int triangleIndex;

    SweepEvent(float pos, Type t, int idx) : position(pos), type(t), triangleIndex(idx) {}

    bool operator<(const SweepEvent& other) const
    {
        if (position != other.position)
            return position < other.position;
        // Paper ordering: END events before PLANAR before START
        return static_cast<int>(type) < static_cast<int>(other.type);
    }
};

// Kd-tree node structure
struct KdNode
{
    enum NodeType : uint8_t { LEAF, INTERNAL };

    NodeType type;
    Box boundingBox;              // Bounding box for this node

    // For leaf nodes
    std::vector<int> triangleIndices;  // Indices into the global triangle array

    // For internal nodes
    int splitAxis;                // 0=X, 1=Y, 2=Z
    float splitPosition;          // Position along the split axis
    KdNode* leftChild;            // Child with smaller coordinates
    KdNode* rightChild;           // Child with larger coordinates
    int depth;                    // Depth in tree for visualization

    // Constructor for leaf node
    KdNode(const Box& bbox, const std::vector<int>& triangles, int d = 0);

    // Constructor for internal node
    KdNode(const Box& bbox, int axis, float pos, KdNode* left, KdNode* right, int d = 0);

    // Destructor
    ~KdNode();

    // Check if a triangle intersects this node's bounding box
    bool TriangleIntersectsBox(const TriangleData& tri) const;
};

class StudentCode
{
public:
    // Debug visualization controls
    bool drawDebugSegs = false;
    bool drawDebugTriangles = false;
    bool drawDebugBoxes = false;
    bool drawDebugSpheres = false;

    // Kd-tree visualization controls
    bool drawKdBoxes = false;
    int kdDisplayLevel = 0;       // Which level to display (0 = all levels)
    int maxKdLevels = 0;          // Maximum depth of the Kd-tree

    // Debug geometry
    std::vector<Segment*> debugSegs;
    std::vector<Triangle*> debugTriangles;
    std::vector<Box*> debugBoxes;
    std::vector<Sphere*> debugSpheres;

    // Scene data
    mat4 ViewInverse, ProjInverse;
    std::vector<TriangleData> triangles;    // All triangles with object references
    std::vector<Object*> objects;

    // Kd-tree data
    KdNode* kdRoot;
    int leafNodeCount;
    int minDepth, maxDepth;
    int rayBoxIntersectionCount;
    int frameCount;
    float avgRayBoxIntersections;

    // SAH parameters
    float CT = 1.5f;              // Cost of traversal
    float CI = 1.0f;              // Cost of intersection
    int maxTrianglesPerLeaf = 8;  // Termination criterion (smaller = deeper tree)
    int maxDepthLimit = 25;       // Depth limit

    // Debug and performance controls
    bool useKdTreeForRayTracing = true;  // Set to false to test linear search performance

    // Triangle selection
    TriangleData* selectedTriangle;

    StudentCode();
    ~StudentCode();

    void SetTransforms(mat4& _vi, mat4& _pi);
    void DrawGui();
    void GenerateRayTracedImage(int width, int height, float* image, vec3& lightPos);
    void Destroy();

    // Collision detection
    bool IsStepLegal(vec3 eye, vec3& direction, float step);
    bool IsStepLegalKd(const Segment& movementSegment, KdNode* node);

    // Triangle management
    void ObjectTriangles(std::vector<vec4>& Pnt,
        std::vector<vec3>& Nrm,
        std::vector<ivec3>& Tri,
        mat4& iTr,
        Object* obj);
    void EndOfTriangles();

    // Kd-tree construction (SAH Sweep Algorithm)
    KdNode* BuildKdTree(const std::vector<int>& triangleIndices, const Box& bbox, int depth = 0);

    // SAH Sweep algorithm components
    struct SplitCandidate
    {
        int axis;
        float position;
        float cost;

        SplitCandidate() : axis(-1), position(0), cost(std::numeric_limits<float>::max()) {}
        SplitCandidate(int a, float p, float c) : axis(a), position(p), cost(c) {}
    };

    // Step 1: Gather - Create sweep events for each triangle
    void GatherSweepEvents(const std::vector<int>& triangleIndices, int axis,
        std::vector<SweepEvent>& events);

    // Step 2: Sort - Sort events by position (handled by std::sort)

    // Step 3: Group - Group events by position and count starts/ends
    struct EventGroup
    {
        float position;
        int numStarts;
        int numEnds;
        int numPlanar;

        EventGroup(float pos) : position(pos), numStarts(0), numEnds(0), numPlanar(0) {}
    };
    void GroupEvents(const std::vector<SweepEvent>& events, std::vector<EventGroup>& groups);

    // Step 4: Sweep - Calculate costs incrementally
    float CalculateSAHCost(const Box& bbox, int axis, float splitPos, int numLeft, int numRight);
    SplitCandidate FindBestSplit(const std::vector<int>& triangleIndices, const Box& bbox);

    // Step 5: Termination decision
    bool ShouldTerminate(const std::vector<int>& triangleIndices, const Box& bbox,
        const SplitCandidate& bestSplit, int depth);

    // Step 6: Split plane decision and triangle distribution
    void SplitTriangles(const std::vector<int>& triangleIndices, int axis, float splitPos,
        std::vector<int>& leftTriangles, std::vector<int>& rightTriangles);

    // Kd-tree traversal and queries
    TriangleData* TraverseKdTreeForRay(const Ray& ray, KdNode* node, float& closestT);
    TriangleData* SelectTriangleAtScreenCenter();

    // Statistics and visualization
    void CalculateKdTreeStatistics(KdNode* node, int depth = 0);
    void DrawKdTreeLevel(unsigned int programId, KdNode* node, int currentLevel, int targetLevel, vec3 color);

    // Debug drawing
    void DrawDebug(unsigned int programId);
    void DrawDebug(unsigned int programId, Segment* seg, vec3 color = vec3(1, 1, 0));
    void DrawDebug(unsigned int programId, Triangle* tri, vec3 color = vec3(1, 1, 0));
    void DrawDebug(unsigned int programId, Box* box, vec3 color = vec3(1, 1, 0));
    void DrawDebug(unsigned int programId, Sphere* sphere, vec3 color = vec3(1, 1, 0));

private:
    // Helper functions
    Box CalculateBoundingBox(const std::vector<int>& triangleIndices);
    Box SplitBox(const Box& bbox, int axis, float splitPos, bool leftSide);
    float CalculateSurfaceArea(const Box& bbox);
};

#endif