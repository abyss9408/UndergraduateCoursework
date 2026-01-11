#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <limits>

#include "imgui.h"

#include <glbinding/gl/gl.h>
#include <glbinding/Binding.h>
using namespace gl;

#include <glu.h>                // For gluErrorString

#include "shader.h"
#include "object.h"
#include "student_code.h"
#include "transform.h"
#include <stdio.h>

vec3 Hdiv(const vec4& H)
{
    return H.xyz() / H.w;
}

// Helper function for 3x3 determinant calculation (for barycentric coordinates)
float det3(const vec3& a, const vec3& b, const vec3& c)
{
    return a.x * (b.y * c.z - b.z * c.y)
         + a.y * (b.z * c.x - b.x * c.z)
         + a.z * (b.x * c.y - b.y * c.x);
}

////////////////////////////////////////////////////////////////////////
// This macro makes it easy to sprinkle checks for OpenGL errors
// throughout your code.  Most OpenGL calls can record errors, and a
// careful programmer will check the error status *often*, perhaps as
// often as after every OpenGL call.  At the very least, once per
// refresh will tell you if something is going wrong.
#define CHECKERROR {GLenum err = glGetError(); if (err != GL_NO_ERROR) { fprintf(stderr, "OpenGL error (at line scene.cpp:%d): %s\n", __LINE__, gluErrorString(err)); exit(-1);} }

// Debug VAO creation (same as before)
static unsigned int segmentVao;
static unsigned int triangleVao;
static unsigned int boxVao;
static unsigned int sphereVao;

void Setup(int programId, mat4& mtr, vec3& color)
{
    const int debugId = 5;

    int loc = glGetUniformLocation(programId, "objectId");
    glUniform1i(loc, debugId);

    loc = glGetUniformLocation(programId, "diffuse");
    glUniform3fv(loc, 1, &color[0]);

    loc = glGetUniformLocation(programId, "ModelTr");
    glUniformMatrix4fv(loc, 1, GL_FALSE, Pntr(mtr));
}

unsigned int VaoFromPoints(std::vector<vec4> Pnt, std::vector<int> Ind)
{
    CHECKERROR;
    unsigned int vaoID;
    glGenVertexArrays(1, &vaoID);
    glBindVertexArray(vaoID);

    GLuint Pbuff;
    glGenBuffers(1, &Pbuff);
    glBindBuffer(GL_ARRAY_BUFFER, Pbuff);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * Pnt.size(),
        &Pnt[0][0], GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    GLuint Ibuff;
    glGenBuffers(1, &Ibuff);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, Ibuff);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * Ind.size(),
        &Ind[0], GL_STATIC_DRAW);

    glBindVertexArray(0);
    CHECKERROR;
    return vaoID;
}

////////////////////////////////////////////////////////////////////////
// KdNode Implementation
////////////////////////////////////////////////////////////////////////

KdNode::KdNode(const Box& bbox, const std::vector<int>& triangles, int d)
    : type(LEAF), boundingBox(bbox), triangleIndices(triangles),
    splitAxis(-1), splitPosition(0), leftChild(nullptr), rightChild(nullptr), depth(d)
{
}

KdNode::KdNode(const Box& bbox, int axis, float pos, KdNode* left, KdNode* right, int d)
    : type(INTERNAL), boundingBox(bbox), splitAxis(axis), splitPosition(pos),
    leftChild(left), rightChild(right), depth(d)
{
}

KdNode::~KdNode()
{
    delete leftChild;
    delete rightChild;
}

bool KdNode::TriangleIntersectsBox(const TriangleData& tri) const
{
    // Simple AABB-triangle intersection test
    // Get triangle bounds
    const Triangle& t = tri.triangle;
    vec3 minBounds = min(min(t[0], t[1]), t[2]);
    vec3 maxBounds = max(max(t[0], t[1]), t[2]);

    // Get box bounds
    vec3 boxMin = boundingBox.center - boundingBox.extents;
    vec3 boxMax = boundingBox.center + boundingBox.extents;

    // Check for overlap
    return !(maxBounds.x < boxMin.x || minBounds.x > boxMax.x ||
        maxBounds.y < boxMin.y || minBounds.y > boxMax.y ||
        maxBounds.z < boxMin.z || minBounds.z > boxMax.z);
}

////////////////////////////////////////////////////////////////////////
// StudentCode Implementation
////////////////////////////////////////////////////////////////////////

StudentCode::StudentCode()
    : kdRoot(nullptr), leafNodeCount(0), minDepth(0), maxDepth(0),
    rayBoxIntersectionCount(0), frameCount(0), avgRayBoxIntersections(0.0f),
    selectedTriangle(nullptr)
{
    CHECKERROR;

    // @@ Example debug for Segments.  (Needs Triangles, Boxes, and Spheres.)
    // This creates a VAO for a generic segment from (0,0,0) to (1,1,1).
    std::vector<vec4> segVerexList = { vec4(0,0,0,1), vec4(1,1,1,1) };
    std::vector<int> segIndexList = { 0,1 };
    segmentVao = VaoFromPoints(segVerexList, segIndexList);

    // Triangle VAO (unit triangle)
    std::vector<vec4> triVertexList =
    {
        vec4(0,0,0,1), vec4(1,0,0,1), vec4(0,1,0,1)
    };
    std::vector<int> triIndexList = { 0, 1, 2 };  // Wireframe edges
    triangleVao = VaoFromPoints(triVertexList, triIndexList);

    // Box VAO (unit cube wireframe)
    std::vector<vec4> boxVertexList =
    {
        // Bottom face
        vec4(-1,-1,-1,1), vec4(1,-1,-1,1), vec4(1,1,-1,1), vec4(-1,1,-1,1),
        // Top face  
        vec4(-1,-1,1,1), vec4(1,-1,1,1), vec4(1,1,1,1), vec4(-1,1,1,1)
    };
    std::vector<int> boxIndexList =
    {
        // Bottom face
        0,1, 1,2, 2,3, 3,0,
        // Top face
        4,5, 5,6, 6,7, 7,4,
        // Vertical edges
        0,4, 1,5, 2,6, 3,7
    };
    boxVao = VaoFromPoints(boxVertexList, boxIndexList);

    // Create sphere VAO (wireframe circles)
    std::vector<vec4> sphereVertexList;
    std::vector<int> sphereIndexList;

    // Create vertices for a sphere wireframe
    int segments = 64;

    // XY circle (around Z axis)
    for (int i = 0; i < segments; ++i)
    {
        float angle = static_cast<float>(2.0f * M_PI * i) / segments;
        sphereVertexList.push_back(vec4(cos(angle), sin(angle), 0, 1));
    }

    // XZ circle (around Y axis) 
    for (int i = 0; i < segments; ++i)
    {
        float angle = static_cast<float>(2.0f * M_PI * i) / segments;
        sphereVertexList.push_back(vec4(cos(angle), 0, sin(angle), 1));
    }

    // YZ circle (around X axis)
    for (int i = 0; i < segments; ++i)
    {
        float angle = static_cast<float>(2.0f * M_PI * i) / segments;
        sphereVertexList.push_back(vec4(0, cos(angle), sin(angle), 1));
    }

    // Silhouette circle (diagonal plane for better 3D appearance)
    for (int i = 0; i < segments; ++i)
    {
        float angle = static_cast<float>(2.0f * M_PI * i) / segments;
        float x = cos(angle) * 0.707f;  // cos(45�) 
        float y = sin(angle) * 0.707f;  // sin(45�)
        float z = cos(angle) * 0.707f;  // Creates a tilted circle
        sphereVertexList.push_back(vec4(x, y, z, 1));
    }

    // Create indices for the 4 circles
    for (int circle = 0; circle < 4; ++circle)
    {
        int offset = circle * segments;
        for (int i = 0; i < segments; ++i)
        {
            sphereIndexList.push_back(offset + i);
            sphereIndexList.push_back(offset + (i + 1) % segments);
        }
    }

    sphereVao = VaoFromPoints(sphereVertexList, sphereIndexList);
}

StudentCode::~StudentCode()
{
    delete kdRoot;
}

void StudentCode::SetTransforms(mat4& _vi, mat4& _pi)
{
    ViewInverse = _vi;
    ProjInverse = _pi;
}

bool StudentCode::IsStepLegal(vec3 eye, vec3& direction, float step)
{
    vec3 endPoint = eye + direction * step;
    Segment movementSegment(eye, endPoint);

    if (frameCount == 0)
    {
        rayBoxIntersectionCount = 0;
    }

    return IsStepLegalKd(movementSegment, kdRoot);
}

bool StudentCode::IsStepLegalKd(const Segment& movementSegment, KdNode* node)
{
    if (!node) return true; // No collision

    ++rayBoxIntersectionCount;

    // Test segment against node's AABB (simple overlap test)
    vec3 segMin = min(movementSegment.point1, movementSegment.point2);
    vec3 segMax = max(movementSegment.point1, movementSegment.point2);

    vec3 nodeMin = node->boundingBox.center - node->boundingBox.extents;
    vec3 nodeMax = node->boundingBox.center + node->boundingBox.extents;

    if (segMax.x < nodeMin.x || segMin.x > nodeMax.x ||
        segMax.y < nodeMin.y || segMin.y > nodeMax.y ||
        segMax.z < nodeMin.z || segMin.z > nodeMax.z)
    {
        return true; // No overlap, no collision possible
    }

    if (node->type == KdNode::LEAF)
    {
        // Test segment against all triangles in leaf
        for (int idx : node->triangleIndices)
        {
            float t;
            if (Intersects(movementSegment, triangles[idx].triangle, &t))
            {
                return false; // Collision detected
            }
        }
        return true; // No collision in this leaf
    }
    else
    {
        // Internal node: test children based on split plane
        bool leftLegal = IsStepLegalKd(movementSegment, node->leftChild);
        if (!leftLegal) return false;

        bool rightLegal = IsStepLegalKd(movementSegment, node->rightChild);
        return rightLegal;
    }
}

void StudentCode::ObjectTriangles(std::vector<vec4>& Pnt,
    std::vector<vec3>& Nrm,
    std::vector<ivec3>& Tri,
    mat4& iTr,
    Object* obj)
{
    objects.push_back(obj);

    // Transform the normal matrix (inverse transpose of model matrix)
    mat3 normalMatrix = transpose(inverse(mat3(iTr)));

    for (const ivec3& tri : Tri)
    {
        vec3 P0 = (iTr * Pnt[tri[0]]).xyz();
        vec3 P1 = (iTr * Pnt[tri[1]]).xyz();
        vec3 P2 = (iTr * Pnt[tri[2]]).xyz();

        Triangle triangle(P0, P1, P2);
        
        // Check if vertex normals are available and indices are valid
        if (!Nrm.empty() && 
            tri[0] >= 0 && tri[0] < static_cast<int>(Nrm.size()) && 
            tri[1] >= 0 && tri[1] < static_cast<int>(Nrm.size()) && 
            tri[2] >= 0 && tri[2] < static_cast<int>(Nrm.size()))
        {
            // Transform and normalize the vertex normals
            vec3 N0 = normalize(normalMatrix * Nrm[tri[0]]);
            vec3 N1 = normalize(normalMatrix * Nrm[tri[1]]);
            vec3 N2 = normalize(normalMatrix * Nrm[tri[2]]);
            
            triangles.emplace_back(triangle, obj, N0, N1, N2);
        }
        else
        {
            triangles.emplace_back(triangle, obj);
        }
    }
}

void StudentCode::EndOfTriangles()
{
    // Clear previous Kd-tree
    delete kdRoot;
    kdRoot = nullptr;

    printf("Building Kd-tree for %ld triangles...\n", triangles.size());

    if (!triangles.empty())
    {
        // Create index list for all triangles
        std::vector<int> allTriangles;
        allTriangles.reserve(triangles.size());
        for (size_t i = 0; i < triangles.size(); ++i)
        {
            allTriangles.push_back(static_cast<int>(i));
        }

        // Calculate overall bounding box
        Box overallBBox = CalculateBoundingBox(allTriangles);

        // Build the Kd-tree
        kdRoot = BuildKdTree(allTriangles, overallBBox, 0);

        // Calculate statistics
        leafNodeCount = 0;
        minDepth = maxDepth = 0;
        maxKdLevels = 0;
        CalculateKdTreeStatistics(kdRoot);

        printf("Kd-tree built successfully!\n");
        printf("Leaf nodes: %d\n", leafNodeCount);
        printf("Min depth: %d, Max depth: %d\n", minDepth, maxDepth);
        printf("Expected depth: ~%.1f\n", leafNodeCount > 0 ? log2f(float(leafNodeCount)) : 0.0f);
    }

    // Clear debug objects and create new ones
    Destroy();

    // Create debug visualizations
    for (size_t i = 0; i < triangles.size(); ++i)
    {
        const Triangle& tri = triangles[i].triangle;
        vec3 center = (tri[0] + tri[1] + tri[2]) / 3.0f;
        vec3 edge1 = tri[1] - tri[0];
        vec3 edge2 = tri[2] - tri[0];
        vec3 normal = normalize(cross(edge1, edge2));
        vec3 normalEnd = center + normal * 0.25f;
        debugSegs.push_back(new Segment(center, normalEnd));
        debugTriangles.push_back(new Triangle(tri));
    }

    // Per-object bounding boxes
    for (Object* obj : objects)
    {
        if (!obj) continue;

        vec3 minBounds(FLT_MAX);
        vec3 maxBounds(-FLT_MAX);
        bool hasTriangles = false;

        for (const TriangleData& triData : triangles)
        {
            if (triData.obj == obj)
            {
                hasTriangles = true;
                for (const vec3& point : triData.triangle.points)
                {
                    minBounds = min(minBounds, point);
                    maxBounds = max(maxBounds, point);
                }
            }
        }

        if (hasTriangles)
        {
            vec3 center = (minBounds + maxBounds) * 0.5f;
            vec3 extents = (maxBounds - minBounds) * 0.5f;
            debugBoxes.push_back(new Box(center, extents));

            float radius = length(extents);
            debugSpheres.push_back(new Sphere(center, radius));
        }
    }
}

////////////////////////////////////////////////////////////////////////
// Kd-tree Construction - SAH Sweep Algorithm
////////////////////////////////////////////////////////////////////////

KdNode* StudentCode::BuildKdTree(const std::vector<int>& triangleIndices, const Box& bbox, int depth)
{
    // Step 5: Termination decision
    if (triangleIndices.size() <= maxTrianglesPerLeaf || depth >= maxDepthLimit) {
        return new KdNode(bbox, triangleIndices, depth);
    }

    // Step 6: Find best split
    SplitCandidate bestSplit = FindBestSplit(triangleIndices, bbox);

    // Check if we should terminate based on SAH cost
    if (ShouldTerminate(triangleIndices, bbox, bestSplit, depth))
    {
        return new KdNode(bbox, triangleIndices, depth);
    }

    // Split triangles
    std::vector<int> leftTriangles, rightTriangles;
    SplitTriangles(triangleIndices, bestSplit.axis, bestSplit.position, leftTriangles, rightTriangles);

    // Safety check: ensure we don't create empty children (except in degenerate cases)
    if (leftTriangles.empty() || rightTriangles.empty())
    {
        return new KdNode(bbox, triangleIndices, depth);
    }

    // Create child bounding boxes
    Box leftBox = SplitBox(bbox, bestSplit.axis, bestSplit.position, true);
    Box rightBox = SplitBox(bbox, bestSplit.axis, bestSplit.position, false);

    // Recursively build children
    KdNode* leftChild = BuildKdTree(leftTriangles, leftBox, depth + 1);
    KdNode* rightChild = BuildKdTree(rightTriangles, rightBox, depth + 1);

    return new KdNode(bbox, bestSplit.axis, bestSplit.position, leftChild, rightChild, depth);
}

// Step 1: Gather - Create sweep events for triangles
void StudentCode::GatherSweepEvents(const std::vector<int>& triangleIndices, int axis,
    std::vector<SweepEvent>& events)
{
    events.clear();
    events.reserve(triangleIndices.size() * 2); // May need more for planar

    const float epsilon = 1e-6f;

    for (int idx : triangleIndices)
    {
        const Triangle& tri = triangles[idx].triangle;

        float minPos = tri[0][axis];
        float maxPos = tri[0][axis];

        for (int i = 1; i < 3; ++i)
        {
            minPos = std::min(minPos, tri[i][axis]);
            maxPos = std::max(maxPos, tri[i][axis]);
        }

        // Check if triangle is planar (degenerate along this axis)
        if (abs(maxPos - minPos) < epsilon)
        {
            events.emplace_back(minPos, SweepEvent::PLANAR, idx);
        }
        else
        {
            events.emplace_back(minPos, SweepEvent::START, idx);
            events.emplace_back(maxPos, SweepEvent::END, idx);
        }
    }
}

// Step 3: Group - Group events by position
void StudentCode::GroupEvents(const std::vector<SweepEvent>& events, std::vector<EventGroup>& groups)
{
    groups.clear();
    if (events.empty()) return;

    groups.emplace_back(events[0].position);

    for (const SweepEvent& event : events)
    {
        if (std::abs(event.position - groups.back().position) > 1e-6f)
        {
            groups.emplace_back(event.position);
        }

        EventGroup& group = groups.back();
        switch (event.type)
        {
            case SweepEvent::START:  ++group.numStarts; break;
            case SweepEvent::END:    ++group.numEnds; break;
            case SweepEvent::PLANAR: ++group.numPlanar; break;
        }
    }
}

// Calculate SAH cost for a potential split
float StudentCode::CalculateSAHCost(const Box& bbox, int axis, float splitPos, int numLeft, int numRight)
{
    Box leftBox = SplitBox(bbox, axis, splitPos, true);
    Box rightBox = SplitBox(bbox, axis, splitPos, false);

    float totalSA = CalculateSurfaceArea(bbox);
    float leftSA = CalculateSurfaceArea(leftBox);
    float rightSA = CalculateSurfaceArea(rightBox);

    float probLeft = leftSA / totalSA;
    float probRight = rightSA / totalSA;

    return CT + probLeft * numLeft * CI + probRight * numRight * CI;
}

// Step 4: Sweep - Find best split using SAH
StudentCode::SplitCandidate StudentCode::FindBestSplit(const std::vector<int>& triangleIndices, const Box& bbox)
{
    SplitCandidate bestSplit;

    for (int axis = 0; axis < 3; ++axis)
    {
        // Step 1: Gather
        std::vector<SweepEvent> events;
        GatherSweepEvents(triangleIndices, axis, events);

        if (events.empty()) continue;

        // Step 2: Sort
        std::sort(events.begin(), events.end());

        // Step 3: Group
        std::vector<EventGroup> groups;
        GroupEvents(events, groups);

        // Step 4: Sweep
        int NL = 0; // Number of triangles on left
        int NR = static_cast<int>(triangleIndices.size()); // Number of triangles on right
        int NP = 0;  // Number of triangles in the plane

        // Remove initial planar triangles from NR
        if (!groups.empty())
        {
            NR -= groups[0].numPlanar;
            NP = groups[0].numPlanar;
        }

        for (size_t i = 0; i < groups.size(); ++i)
        {
            const EventGroup& group = groups[i];

            // Update counters following paper exactly (page 9)
            if (i > 0)
            {
                // NL += pk-1^0 + pk-1^+
                NL += groups[i - 1].numPlanar + groups[i - 1].numStarts;
            }

            // NR -= pk^0 + pk^-
            NR -= group.numPlanar + group.numEnds;
            NP = group.numPlanar;

            // Calculate cost at this position
            // |TL| = NL + NP, |TR| = NR + NP (triangles in plane go to both sides)
            float cost = CalculateSAHCost(bbox, axis, group.position, NL + NP, NR + NP);

            if (cost < bestSplit.cost)
            {
                bestSplit = SplitCandidate(axis, group.position, cost);
            }
        }
    }

    return bestSplit;
}

// Step 5: Termination decision
bool StudentCode::ShouldTerminate(const std::vector<int>& triangleIndices, const Box& bbox,
    const SplitCandidate& bestSplit, int depth)
{
    // Basic termination criteria
    if (triangleIndices.size() <= maxTrianglesPerLeaf) return true;
    if (depth >= maxDepthLimit) return true;
    if (bestSplit.axis == -1) return true; // No valid split found

    // Cost of not splitting (making this a leaf)
    float leafCost = CI * static_cast<float>(triangleIndices.size());

    // SAH termination: only split if it reduces cost
    if (bestSplit.cost >= leafCost) return true;

    // Additional check: don't split if it doesn't significantly reduce triangle count
    // This helps prevent excessive duplication
    std::vector<int> leftTriangles, rightTriangles;
    SplitTriangles(triangleIndices, bestSplit.axis, bestSplit.position, leftTriangles, rightTriangles);

    // If split doesn't reduce triangle count meaningfully, terminate
    float reductionRatio = static_cast<float>(leftTriangles.size() + rightTriangles.size()) /
        static_cast<float>(triangleIndices.size());

    if (reductionRatio > 1.5f)
    {
        // Too much duplication
        return true;
    }

    return false;
}

// Step 6: Split triangles based on split plane
void StudentCode::SplitTriangles(const std::vector<int>& triangleIndices, int axis, float splitPos,
    std::vector<int>& leftTriangles, std::vector<int>& rightTriangles)
{
    leftTriangles.clear();
    rightTriangles.clear();

    for (int idx : triangleIndices)
    {
        const Triangle& tri = triangles[idx].triangle;

        // Find min and max along the axis
        float minPos = tri[0][axis];
        float maxPos = tri[0][axis];

        for (int i = 1; i < 3; ++i)
        {
            minPos = std::min(minPos, tri[i][axis]);
            maxPos = std::max(maxPos, tri[i][axis]);
        }

        // Classify triangle based on split plane
        bool leftSide = minPos < splitPos;
        bool rightSide = maxPos > splitPos;
        bool crossing = leftSide && rightSide;

        // More conservative splitting to reduce duplication
        if (crossing)
        {
            // For crossing triangles, add to both sides
            leftTriangles.push_back(idx);
            rightTriangles.push_back(idx);
        }
        else if (leftSide)
        {
            // Triangle is entirely on left side
            leftTriangles.push_back(idx);
        }
        else if (rightSide)
        {
            // Triangle is entirely on right side  
            rightTriangles.push_back(idx);
        }
        else
        {
            // Triangle is exactly on the plane - add to left side by convention
            leftTriangles.push_back(idx);
        }
    }
}

////////////////////////////////////////////////////////////////////////
// Helper Functions
////////////////////////////////////////////////////////////////////////

Box StudentCode::CalculateBoundingBox(const std::vector<int>& triangleIndices)
{
    if (triangleIndices.empty())
    {
        return Box(vec3(0), vec3(0));
    }

    vec3 minBounds = triangles[triangleIndices[0]].triangle[0];
    vec3 maxBounds = triangles[triangleIndices[0]].triangle[0];

    for (int idx : triangleIndices)
    {
        const Triangle& tri = triangles[idx].triangle;
        for (const vec3& point : tri.points)
        {
            minBounds = min(minBounds, point);
            maxBounds = max(maxBounds, point);
        }
    }

    vec3 center = (minBounds + maxBounds) * 0.5f;
    vec3 extents = (maxBounds - minBounds) * 0.5f;
    return Box(center, extents);
}

Box StudentCode::SplitBox(const Box& bbox, int axis, float splitPos, bool leftSide)
{
    vec3 minBounds = bbox.center - bbox.extents;
    vec3 maxBounds = bbox.center + bbox.extents;

    if (leftSide)
    {
        maxBounds[axis] = splitPos;
    }
    else
    {
        minBounds[axis] = splitPos;
    }

    vec3 center = (minBounds + maxBounds) * 0.5f;
    vec3 extents = (maxBounds - minBounds) * 0.5f;
    return Box(center, extents);
}

float StudentCode::CalculateSurfaceArea(const Box& bbox)
{
    vec3 size = bbox.extents * 2.0f;
    return 2.0f * (size.x * size.y + size.y * size.z + size.z * size.x);
}

////////////////////////////////////////////////////////////////////////
// Ray Traversal and Triangle Selection
////////////////////////////////////////////////////////////////////////

TriangleData* StudentCode::TraverseKdTreeForRay(const Ray& ray, KdNode* node, float& closestT)
{
    if (!node) return nullptr;

    ++rayBoxIntersectionCount;

    // Test ray against bounding box
    float boxT;
    if (!Intersects(ray, node->boundingBox, &boxT) || boxT >= closestT)
    {
        return nullptr;
    }

    if (node->type == KdNode::LEAF)
    {
        TriangleData* closestTriangle = nullptr;

        for (int idx : node->triangleIndices)
        {
            float t, u, v;
            if (Intersects(ray, triangles[idx].triangle, &t, &u, &v))
            {
                if (t > 0.0f && t < closestT)
                {
                    closestT = t;
                    closestTriangle = &triangles[idx];
                }
            }
        }
        return closestTriangle;
    }
    else
    {
        // Internal node: proper Kd-tree traversal
        int axis = node->splitAxis;
        float splitPos = node->splitPosition;

        // Determine ray direction and origin along split axis
        float rayOrigin = ray.origin[axis];
        float rayDir = ray.direction[axis];

        // Determine which child to visit first
        KdNode* nearChild, * farChild;
        if (rayOrigin < splitPos)
        {
            nearChild = node->leftChild;
            farChild = node->rightChild;
        }
        else
        {
            nearChild = node->rightChild;
            farChild = node->leftChild;
        }

        // Traverse near child first
        TriangleData* result = nullptr;
        if (nearChild)
        {
            result = TraverseKdTreeForRay(ray, nearChild, closestT);
        }

        if (farChild)
        {
            // Calculate intersection with split plane
            if (abs(rayDir) > 1e-6f)
            {
                float t = (splitPos - rayOrigin) / rayDir;
                // Only traverse far child if split plane is closer than current best hit
                if (t >= 0.0f && t < closestT)
                {
                    TriangleData* farResult = TraverseKdTreeForRay(ray, farChild, closestT);
                    if (farResult)
                    {
                        result = farResult;
                    }
                }
            }
            else
            {
                // Ray is parallel to split plane, traverse far child anyway
                TriangleData* farResult = TraverseKdTreeForRay(ray, farChild, closestT);
                if (farResult)
                {
                    result = farResult;
                }
            }
        }

        return result;
    }
}

TriangleData* StudentCode::SelectTriangleAtScreenCenter()
{
    vec3 eye = Hdiv(ViewInverse * vec4(0, 0, 0, 1));
    vec3 T = Hdiv(ViewInverse * ProjInverse * vec4(0, 0, -1, 1));
    Ray ray(eye, T - eye);

    float closestDistance = std::numeric_limits<float>::max();
    return TraverseKdTreeForRay(ray, kdRoot, closestDistance);
}

////////////////////////////////////////////////////////////////////////
// Statistics and Visualization
////////////////////////////////////////////////////////////////////////

void StudentCode::CalculateKdTreeStatistics(KdNode* node, int depth)
{
    if (!node) return;

    if (node->type == KdNode::LEAF)
    {
        ++leafNodeCount;
        if (leafNodeCount == 1)
        {
            minDepth = maxDepth = depth;
        }
        else
        {
            minDepth = std::min(minDepth, depth);
            maxDepth = std::max(maxDepth, depth);
        }
        maxKdLevels = std::max(maxKdLevels, depth);
    }
    else
    {
        CalculateKdTreeStatistics(node->leftChild, depth + 1);
        CalculateKdTreeStatistics(node->rightChild, depth + 1);
    }
}

void StudentCode::DrawKdTreeLevel(unsigned int programId, KdNode* node, int currentLevel, int targetLevel, vec3 color)
{
    if (!node || triangles.size() > 1000000) return;

    if (targetLevel == 0 || currentLevel == targetLevel)
    {
        // Draw with shrinking based on depth (as specified in project requirements)
        Box shrunkBox = node->boundingBox;
        float shrinkFactor = powf(0.99f, static_cast<float>(node->depth));
        shrunkBox.extents *= shrinkFactor;

        DrawDebug(programId, &shrunkBox, color);
    }

    if (node->type == KdNode::INTERNAL && (targetLevel == 0 || currentLevel < targetLevel))
    {
        vec3 leftColor = vec3(1.0f - currentLevel * 0.1f, currentLevel * 0.1f, 0.5f);
        vec3 rightColor = vec3(0.5f, 1.0f - currentLevel * 0.1f, currentLevel * 0.1f);

        DrawKdTreeLevel(programId, node->leftChild, currentLevel + 1, targetLevel, leftColor);
        DrawKdTreeLevel(programId, node->rightChild, currentLevel + 1, targetLevel, rightColor);
    }
}

void StudentCode::Destroy()
{
    for (Segment* seg : debugSegs) delete seg;
    for (Triangle* tri : debugTriangles) delete tri;
    for (Box* box : debugBoxes) delete box;
    for (Sphere* sphere : debugSpheres) delete sphere;

    debugSegs.clear();
    debugTriangles.clear();
    debugBoxes.clear();
    debugSpheres.clear();
}

void StudentCode::DrawDebug(unsigned int programId)
{
    selectedTriangle = SelectTriangleAtScreenCenter();

    // Draw Kd-tree boxes if enabled
    if (drawKdBoxes && kdRoot)
    {
        vec3 kdColor = vec3(0, 1, 1); // Cyan for Kd-tree boxes
        DrawKdTreeLevel(programId, kdRoot, 0, kdDisplayLevel, kdColor);
    }

    // Draw selected triangle in bright color
    if (selectedTriangle)
    {
        DrawDebug(programId, &selectedTriangle->triangle, vec3(1, 1, 0)); // Bright yellow
    }

    // Draw debug objects
    if (drawDebugSegs) for (Segment* seg : debugSegs) DrawDebug(programId, seg);
    if (drawDebugTriangles) for (Triangle* tri : debugTriangles) DrawDebug(programId, tri);
    if (drawDebugBoxes) for (Box* box : debugBoxes) DrawDebug(programId, box);
    if (drawDebugSpheres) for (Sphere* sphere : debugSpheres) DrawDebug(programId, sphere);
}

// Debug drawing functions (same as before)
void StudentCode::DrawDebug(unsigned int programId, Segment* seg, vec3 color)
{
    vec3& A = seg->point1;
    vec3& B = seg->point2;
    mat4 mtr = Translate(A) * Scale(B - A);

    Setup(programId, mtr, color);
    glBindVertexArray(segmentVao);
    glDrawElements(GL_LINES, 2, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void StudentCode::DrawDebug(unsigned int programId, Triangle* tri, vec3 color)
{
    vec3& A = tri->points[0];
    vec3& B = tri->points[1];
    vec3& C = tri->points[2];

    vec3 AB = B - A;
    vec3 AC = C - A;

    mat4 mtr = mat4(
        AB.x, AB.y, AB.z, 0.0f,
        AC.x, AC.y, AC.z, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        A.x, A.y, A.z, 1.0f
    );

    Setup(programId, mtr, color);
    glBindVertexArray(triangleVao);
    glDrawElements(GL_LINE_LOOP, 3, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void StudentCode::DrawDebug(unsigned int programId, Box* box, vec3 color)
{
    mat4 mtr = Translate(box->center) * Scale(box->extents);

    Setup(programId, mtr, color);
    glBindVertexArray(boxVao);
    glDrawElements(GL_LINES, 24, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void StudentCode::DrawDebug(unsigned int programId, Sphere* sphere, vec3 color)
{
    // Transform unit sphere to sphere position and size
    mat4 mtr = Translate(sphere->center) * Scale(vec3(sphere->radius));

    Setup(programId, mtr, color);
    glBindVertexArray(sphereVao);
    // 4 circles * 64 segments * 2 vertices per line = 512 indices
    glDrawElements(GL_LINES, 512, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void StudentCode::DrawGui()
{
    static bool popup = false;
    if (ImGui::Checkbox("Report", &popup)) ImGui::OpenPopup("Report");
    if (ImGui::BeginPopupModal("Report", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
    {
        ImGui::Text("Kd-tree Implementation Report\n\
\n\
SAH Sweep Algorithm Implementation:\n\
Gather: 431 - 463\n\
Sort: 520\n\
Group: 466 - 488\n\
Sweep: 527 - 561\n\
Terminate decision: 568 - 598\n\
Split-plane decision: 601 - 649\n\
\n\
Statistics:\n\
Triangles: %ld\n\
Leaf nodes: %d\n\
Tree depth: %d - %d\n\
Ray tracing without kd tree (100x100 and 100 triangles): 20 fps\n\
Ray tracing with kd tree (100x100 and 100 triangles): 47 fps", triangles.size(), leafNodeCount, minDepth, maxDepth);

        if (ImGui::Button("Close", ImVec2(60, 0))) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }

    ImGui::Checkbox("Draw Segments", &drawDebugSegs);
    ImGui::Checkbox("Draw Triangles", &drawDebugTriangles);
    ImGui::Checkbox("Draw Boxes", &drawDebugBoxes);
    ImGui::Checkbox("Draw Spheres", &drawDebugSpheres);

    ImGui::Separator();

    ImGui::Text("Ray Tracing");
    ImGui::Checkbox("Use Kd-tree for ray tracing", &useKdTreeForRayTracing);
    if (!useKdTreeForRayTracing)
    {
        ImGui::TextColored(ImVec4(1, 1, 0, 1), "Using LINEAR search (slow!)");
    }

    ImGui::Separator();

    ImGui::Text("Kd-tree Visualization");
    ImGui::Checkbox("Draw Kd-tree Boxes", &drawKdBoxes);

    if (maxKdLevels > 0)
    {
        ImGui::SliderInt("Kd-tree Level", &kdDisplayLevel, 0, maxKdLevels);
        if (kdDisplayLevel == 0)
        {
            ImGui::Text("Showing all levels");
        }
        else
        {
            ImGui::Text("Showing level %d", kdDisplayLevel);
        }
    }

    ImGui::Separator();

    ImGui::Text("Kd-tree Statistics");
    ImGui::Text("Triangles: %ld", triangles.size());
    ImGui::Text("Leaf nodes: %d", leafNodeCount);
    ImGui::Text("Tree depth: %d - %d", minDepth, maxDepth);
    ImGui::Text("Expected depth: ~%.1f", leafNodeCount > 0 ? log2f(float(leafNodeCount)) : 0.0f);

    ++frameCount;
    if (frameCount == 1)
    {
        avgRayBoxIntersections = static_cast<float>(rayBoxIntersectionCount);
    }
    else
    {
        avgRayBoxIntersections = (avgRayBoxIntersections * (frameCount - 1) + rayBoxIntersectionCount) / frameCount;
    }

    ImGui::Text("Ray/Box tests this frame: %d", rayBoxIntersectionCount);
    ImGui::Text("Average ray/box tests: %.1f", avgRayBoxIntersections);
    ImGui::Text("Frame count: %d", frameCount);

    rayBoxIntersectionCount = 0;
}

void StudentCode::GenerateRayTracedImage(int width, int height, float* image, vec3& lightPos)
{
    // Ray trace a smaller region in the center for performance
    int N = 100; // Number of pixels to ray-trace on each side of center
    vec3 eye = Hdiv(ViewInverse * vec4(0, 0, 0, 1));

    // Ensure we don't go outside image bounds
    int startI = std::max(0, width / 2 - N);
    int endI = std::min(width - 1, width / 2 + N);
    int startJ = std::max(0, height / 2 - N);
    int endJ = std::min(height - 1, height / 2 + N);

    // Reset ray-box intersection counter for this frame
    rayBoxIntersectionCount = 0;

    for (int i = startI; i <= endI; ++i)
    {
        for (int j = startJ; j <= endJ; ++j)
        {
            // Pixel center converted from (i,j) index to (X,Y,Z) in NDC
            float X = 2.0f * (i + 0.5f) / width - 1.0f;
            float Y = 2.0f * (j + 0.5f) / height - 1.0f;
            vec3 T = Hdiv(ViewInverse * ProjInverse * vec4(X, Y, -1, 1));

            Ray ray(eye, normalize(T - eye));

            // Find closest intersection
            float tMin = std::numeric_limits<float>::max();
            TriangleData* frontTriangle = nullptr;
            float bestU = 0, bestV = 0; // Store barycentric coordinates for best hit

            if (kdRoot && useKdTreeForRayTracing)
            {
                // Use Kd-tree traversal
                frontTriangle = TraverseKdTreeForRay(ray, kdRoot, tMin);
                // Re-compute barycentric coordinates for the hit
                if (frontTriangle)
                {
                    float t, u, v;
                    Intersects(ray, frontTriangle->triangle, &t, &u, &v);
                    bestU = u;
                    bestV = v;
                }
            }
            else
            {
                // Linear search fallback (for comparison/debugging)
                for (size_t idx = 0; idx < triangles.size(); ++idx)
                {
                    float t, u, v;
                    if (Intersects(ray, triangles[idx].triangle, &t, &u, &v))
                    {
                        if (t > 0.0f && t < tMin)
                        {
                            tMin = t;
                            frontTriangle = &triangles[idx];
                            bestU = u;
                            bestV = v;
                        }
                    }
                }
            }

            vec3 color;
            if (frontTriangle)
            {
                // The color of a triangle
                vec3 Kd = frontTriangle->obj->diffuseColor;

                // Triangle vertices
                vec3 P0 = frontTriangle->triangle.points[0];
                vec3 P1 = frontTriangle->triangle.points[1];
                vec3 P2 = frontTriangle->triangle.points[2];
                
                // Calculate intersection point
                vec3 intersectionPoint = ray.origin + tMin * ray.direction;
                
                // Calculate normal - either interpolated or face normal
                vec3 N;
                
                if (frontTriangle->hasVertexNormals)
                {
                    // Normal interpolation using barycentric coordinates
                    // The intersection point P is given by: P = (1-u-v)*P0 + u*P1 + v*P2
                    // where u and v are from the ray-triangle intersection
                    
                    // Calculate barycentric coordinates using determinants
                    float d = det3(P0, P1, P2);
                    float b1 = det3(intersectionPoint, P1, P2) / d;
                    float b2 = det3(P0, intersectionPoint, P2) / d;
                    float b3 = det3(P0, P1, intersectionPoint) / d;
                    
                    // Interpolate normals using barycentric coordinates
                    N = normalize(b1 * frontTriangle->normals[0] + 
                                  b2 * frontTriangle->normals[1] + 
                                  b3 * frontTriangle->normals[2]);
                }
                else
                {
                    // Face normal (not interpolated)
                    N = normalize(cross(P1 - P0, P2 - P0));
                }

                // The direction from the intersection point toward the light
                vec3 L = normalize(lightPos - intersectionPoint);

                // Shadow calculation - check if point is in shadow
                bool inShadow = false;
                
                // Create shadow ray from intersection point toward light
                const float epsilon = 1e-3f; // Offset to avoid self-intersection
                vec3 shadowOrigin = intersectionPoint + epsilon * L;
                vec3 shadowDirection = L;
                float distanceToLight = length(lightPos - intersectionPoint);
                
                Ray shadowRay(shadowOrigin, shadowDirection);
                float shadowT = distanceToLight;
                
                // Check if shadow ray hits anything before reaching the light
                if (kdRoot && useKdTreeForRayTracing)
                {
                    TriangleData* shadowHit = TraverseKdTreeForRay(shadowRay, kdRoot, shadowT);
                    inShadow = (shadowHit != nullptr && shadowT < distanceToLight);
                }
                else
                {
                    // Linear search fallback for shadow ray
                    for (size_t idx = 0; idx < triangles.size(); ++idx)
                    {
                        float t, u, v;
                        if (Intersects(shadowRay, triangles[idx].triangle, &t, &u, &v))
                        {
                            if (t > 0.0f && t < distanceToLight)
                            {
                                inShadow = true;
                                break;
                            }
                        }
                    }
                }

                // BRDF Lighting calculation with shadows
                
                // Material parameters
                vec3 Ks = frontTriangle->obj->specularColor;
                float alpha = frontTriangle->obj->shininess; // roughness parameter
                
                // Light parameters
                vec3 Ii = vec3(3.0f, 3.0f, 3.0f);  // Light intensity
                vec3 Ia = vec3(0.5f, 0.5f, 0.5f);  // Ambient light intensity
                
                // View vector (from intersection point to eye)
                vec3 V = normalize(eye - intersectionPoint);
                
                if (inShadow)
                {
                    // Shadow: ambient only
                    color = Ia * Kd;
                }
                else
                {
                    // Full BRDF lighting calculation
                    
                    // Helper function for clamped dot product
                    auto clampedDot = [](const vec3& a, const vec3& b) {
                        return std::max(0.0f, dot(a, b));
                    };
                    
                    // Half vector
                    vec3 H = normalize(L + V);
                    
                    // Dot products (clamped to non-negative)
                    // Use abs for N dot products to light both sides of polygons
                    float NdotL = std::abs(dot(N, L));
                    float NdotV = std::abs(dot(N, V));
                    float NdotH = std::abs(dot(N, H));
                    float LdotH = clampedDot(L, H);
                    
                    // D(H) - Corrected Phong specular term
                    float D = ((alpha + 2.0f) / (2.0f * M_PI)) * powf(NdotH, alpha);
                    
                    // F(L,H) - Schlick's approximation of Fresnel term
                    vec3 F = Ks + (vec3(1.0f) - Ks) * powf(1.0f - LdotH, 5.0f);
                    
                    // G(L,V,H) - Approximation to shadow/occlusion term (simplified)
                    // Prevent division by zero
                    float LdotH2 = LdotH * LdotH;
                    if (LdotH2 < 1e-6f) LdotH2 = 1e-6f;
                    
                    float G = (NdotL * NdotV) / LdotH2;
                    
                    // BRDF term (using simplified G that cancels some terms)
                    vec3 diffuseTerm = vec3(Kd.x / M_PI, Kd.y / M_PI, Kd.z / M_PI);
                    vec3 specularTerm = (D * F * G) / (4.0f * LdotH2);
                    vec3 BRDF = diffuseTerm + specularTerm;
                    
                    // Final color calculation
                    color = Ia * Kd + Ii * NdotL * BRDF;
                }

                // Clamp color values to prevent artifacts
                color.x = std::min(1.0f, std::max(0.0f, color.x));
                color.y = std::min(1.0f, std::max(0.0f, color.y));
                color.z = std::min(1.0f, std::max(0.0f, color.z));
            }
            else
            {
                color = vec3(0.25f); // Dark gray background
            }

            float* pixel = image + 3 * (j * width + i);
            *pixel++ = color.x;
            *pixel++ = color.y;
            *pixel++ = color.z;
        }
    }
}