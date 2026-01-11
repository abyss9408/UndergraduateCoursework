///////////////////////////////////////////////////////////////////////
// Geometric objects (vec3s, vec3s, Planes, ...) and operations.
////////////////////////////////////////////////////////////////////////
#define _USE_MATH_DEFINES
#include "geomlib.h"
#include <vector>
#include <cassert>
#include <float.h> // FLT_EPSILON
#include <cmath>

////////////////////////////////////////////////////////////////////////
// Distance methods
////////////////////////////////////////////////////////////////////////

// Return the distance from a point to a line.
float Distance(const vec3& point, const Line& line)
{
    // Calculate vector from linePoint to point (P - A)
    const vec3 PA = point - line.point;

    // Calculate cross product (P-A) x v
    const vec3 crossProduct = cross(PA, line.vector);;

    // Calculate magnitude of direction vector ||v||
    const float crossProductMagnitude = sqrtf(dot(crossProduct, crossProduct));

    // Calculate magnitude of direction vector ||v||
    const float directionMagnitude = sqrtf(dot(line.vector, line.vector));

    return crossProductMagnitude / directionMagnitude;
}

// Return the distance from a point to a plane.
float Distance(const vec3& point, const Plane& plane)
{
    const float dotPointNormal = point.x * plane.crds[0] + point.y * plane.crds[1] + point.z * plane.crds[2];

    const float dotPlusDistance = std::fabsf(dotPointNormal + plane.crds[3]);

    const float normalMagnitude = sqrtf(plane.crds[0] * plane.crds[0] + plane.crds[1] * plane.crds[1] + plane.crds[2] * plane.crds[2]);

    return dotPlusDistance / normalMagnitude;
}

////////////////////////////////////////////////////////////////////////
// Containment methods
////////////////////////////////////////////////////////////////////////

// Determines if point (known to be on a line) is contained within a segment.
bool Segment::contains(const vec3& point) const
{
    vec3 AB = point2 - point1;
    vec3 AP = point - point1;

    float lengthSquared = dot(AB, AB);
    float dotProd = dot(AB, AP);

    // Calculate t: how far along AB vector is point
    float t = dotProd / lengthSquared;

    return t >= 0.0f && t <= 1.0f;
}

// Determines if point (known to be on a line) is contained within a ray.
bool Ray::contains(const vec3& point, float *rt) const
{
    vec3 op = point - origin;

    float lengthSquared = dot(direction, direction);
    float dotProd = dot(direction, op);

    // Calculate t
    float t = dotProd / lengthSquared;

    if (t >= 0.f && rt)
    {
        *rt = t;
        return true;
    }

    return false;
}

// Determines if point is contained within a box.
bool Box::contains(const vec3& point) const
{
    if (abs(center.x - point.x) > extents.x || abs(center.y - point.y) > extents.y || abs(center.z - point.z) > extents.z)
    {
        return false;
    }

    return true;
}

// Determines if point (known to be on a plane) is contained within a triangle.
bool Triangle::contains(const vec3& point) const
{
    // Get the three vertices of the triangle
    const vec3& A = points[0];
    const vec3& B = points[1];
    const vec3& C = points[2];
    const vec3& P = point;

    // Calculate the plane normal N = (B-A) × (C-A)
    vec3 AB = B - A;
    vec3 AC = C - A;
    vec3 N = cross(AB, AC);

    // Perform the three edge tests:
    vec3 PA = P - A;
    vec3 cross_N_AB = cross(N, AB);
    float test1 = dot(PA, cross_N_AB);

    vec3 PB = P - B;
    vec3 BC = C - B;
    vec3 cross_N_BC = cross(N, BC);
    float test2 = dot(PB, cross_N_BC);

    vec3 PC = P - C;
    vec3 CA = A - C;
    vec3 cross_N_CA = cross(N, CA);
    float test3 = dot(PC, cross_N_CA);

    // Point is inside if all three tests are satisfied
    return (test1 >= -FLT_EPSILON) && (test2 >= -FLT_EPSILON) && (test3 >= -FLT_EPSILON);
}

////////////////////////////////////////////////////////////////////////
// Intersects functions
// In the following Intersects function these rules apply:
//
// * Most test are to determine if a *unique* solution exists. (Or in
//   some cases up to two intersection points exist.)  Parallel
//   objects have either zero or infinitely many solutions and so
//   return false.
//
// * If a unique solution exists, a function value of true is
//   returned.  (Or in the cases where several solutions can exist,
//   the number of intersection parameters are returned.)
//
// * If a unique solution does exist, the calling program may provide
//   a memory location into which the intersection parameter can be
//   returned.  Such pointer may be NULL to indicate that the
//   intersection parameter is not to be returned.
//
////////////////////////////////////////////////////////////////////////

// Determines if  line and plane have a unique intersection.  
// If true and t is not NULL, returns intersection parameter.
bool Intersects(const Line& line, const Plane& plane, float *rt)
{
    // Get the plane normal vector (A, B, C from Ax + By + Cz + D = 0)
    vec3 normal = plane.normal();

    // Get the line direction vector
    vec3 direction = line.vector;

    // Check if line is parallel to plane
    // If direction · normal = 0, then line is parallel to plane
    float denominator = dot(direction, normal);

    // Use a small epsilon to handle floating point precision
    if (abs(denominator) < FLT_EPSILON)
    {
        // Line is parallel to plane - no unique intersection
        return false;
    }

    float numerator = -(dot(normal, line.point) + plane[3]);
    float t = numerator / denominator;

    // Store the intersection parameter if requested
    if (rt != nullptr)
    {
        *rt = t;
    }

    return true;
}

// Determines if  segment and plane have a unique intersection.  
// If true and rt is not NULL, returns intersection parameter.
bool Intersects(const Segment& seg, const Plane& plane, float *rt)
{
    vec3 segmentVector = seg.point2 - seg.point1;
    Line line(seg.point1, segmentVector);

    // Calculate line-plane intersection using the existing function
    float t;
    bool hasIntersection = Intersects(line, plane, &t);

    if (!hasIntersection)
    {
        // Line is parallel to plane - no intersection
        return false;
    }

    if (t < 0.0f || t > 1.0f)
    {
        // Intersection point is outside the segment
        return false;
    }

    // Store the intersection parameter if requested
    if (rt != nullptr)
    {
        *rt = t;
    }

    return true;
}

// Determines if  ray and triangle have a unique intersection.  
// If true and rt is not NULL, returns intersection parameter.
bool Intersects(const Ray& ray, const Triangle& tri, float *rt, float *ru, float *rv)
{
    vec3 e1 = tri.points[1] - tri.points[0];
    vec3 e2 = tri.points[2] - tri.points[0];
    vec3 p = cross(ray.direction, e2);
    float d = dot(p, e1);
    if (abs(d) < FLT_EPSILON)
    {
        return false;
    }

    vec3 s = ray.origin - tri.points[0];
    float u = dot(p, s) / d;
    if (u < 0.f || u > 1.f)
    {
        return false;
    }

    vec3 q = cross(s, e1);
    float v = dot(ray.direction, q) / d;
    if (v < 0.f || (u + v) > 1.f)
    {
        return false;
    }

    float t = dot(e2, q) / d;
    if (t < 0.f)
    {
        return false;
    }

    if (rt && ru && rv)
    {
        *rt = t;
        *ru = u;
        *rv = v;
    }

    return true;
}

// Determines if  segment and triangle have a unique intersection.  
// If true and rt is not NULL, returns intersection parameter.
bool Intersects(const Segment& seg, const Triangle& tri, float *rt)
{
    vec3 segmentVector = seg.point2 - seg.point1;
    Ray ray(seg.point1, segmentVector);
    
    // Use ray-triangle intersection test
    float t, u, v;
    bool hasIntersection = Intersects(ray, tri, &t, &u, &v);
    
    if (!hasIntersection)
    {
        return false;
    }
    
    // Check if intersection point is within segment bounds
    if (t < 0.0f || t > 1.0f)
    {
        return false;
    }
    
    if (rt)
    {
        *rt = t;
    }
    
    return true;
}

// Determines if  ray and sphere intersect.  
// If so and rt is not NULL, returns intersection parameter.
bool Intersects(const Ray& ray, const Sphere& sphere, float *rt)
{
    float a = dot(ray.direction, ray.direction);
    float b = dot(2.f * (ray.origin - sphere.center), ray.direction);
    float c = dot((ray.origin - sphere.center), (ray.origin - sphere.center)) - (sphere.radius * sphere.radius);
    float discriminant = (b * b) - (4 * a * c);

    // discriminant is zero
    if (abs(discriminant) < FLT_EPSILON)
    {
        if (rt)
        {
            *rt = -b / (2 * a);
        }
        return true;
    }
    else if (discriminant < 0.f)
    {
        return false;
    }
    else if (discriminant > 0.f)
    {
        if (rt)
        {
            *rt = (-b + sqrtf(discriminant)) / (2 * a);
        }
        return true;
    }
}

// Determines if  ray and AABB intersect.  
// If so and rt is not NULL, returns intersection parameter.
bool Intersects(const Ray& ray, const Box& box, float *rt)
{
    float t0 = -std::numeric_limits<float>::infinity();
    float t1 = std::numeric_limits<float>::infinity();

    // x-coordinate
    if (abs(ray.direction.x) < FLT_EPSILON)
    {
        if (ray.origin.x < (box.center.x - box.extents.x) || ray.origin.x > (box.center.x + box.extents.x))
        {
            return false;
        }
    }
    else
    {
        float s0 = ((box.center.x - box.extents.x) - ray.origin.x) / ray.direction.x;
        float s1 = ((box.center.x + box.extents.x) - ray.origin.x) / ray.direction.x;
        if (s0 > s1)
        {
            float temp = s0;
            s0 = s1;
            s1 = temp;
        }
        t0 = std::max(t0, s0);
        t1 = std::min(t1, s1);
    }

    if (t0 > t1 || t1 < 0.f)
    {
        return false;
    }

    // y-coordinate
    if (abs(ray.direction.y) < FLT_EPSILON)
    {
        if (ray.origin.y < (box.center.y - box.extents.y) || ray.origin.y > (box.center.y + box.extents.y))
        {
            return false;
        }
    }
    else
    {
        float s0 = ((box.center.y - box.extents.y) - ray.origin.y) / ray.direction.y;
        float s1 = ((box.center.y + box.extents.y) - ray.origin.y) / ray.direction.y;
        if (s0 > s1)
        {
            float temp = s0;
            s0 = s1;
            s1 = temp;
        }
        t0 = std::max(t0, s0);
        t1 = std::min(t1, s1);
    }

    if (t0 > t1 || t1 < 0.f)
    {
        return false;
    }

    // z-coordinate
    if (abs(ray.direction.z) < FLT_EPSILON)
    {
        if (ray.origin.z < (box.center.z - box.extents.z) || ray.origin.z > (box.center.z + box.extents.z))
        {
            return false;
        }
    }
    else
    {
        float s0 = ((box.center.z - box.extents.z) - ray.origin.z) / ray.direction.z;
        float s1 = ((box.center.z + box.extents.z) - ray.origin.z) / ray.direction.z;
        if (s0 > s1)
        {
            float temp = s0;
            s0 = s1;
            s1 = temp;
        }
        t0 = std::max(t0, s0);
        t1 = std::min(t1, s1);
    }

    if (t0 > t1 || t1 < 0.f)
    {
        return false;
    }
    
    // Set the intersection parameter
    if (rt)
    {
        if (t0 < 0.f)
        {
            *rt = t1;  // Ray origin is inside the box
        }
        else
        {
            *rt = t0;  // Ray origin is outside the box
        }
    }

    return true;
}

// Determines if  triangles intersect.  
// If parallel, returns false. (This may be considered misleading.)
// If true and rpoint is not NULL, returns two edge/triangle intersections.
int Intersects(const Triangle& tri1, const Triangle& tri2,
			   std::pair<vec3, vec3> *rpoints)
{
    std::vector<vec3> intersectionPoints;

    // Check each edge of tri1 against tri2
    for (int i = 0; i < 3; ++i)
    {
        Segment edge1(tri1[i], tri1[(i + 1) % 3]);
        float t;

        if (Intersects(edge1, tri2, &t))
        {
            vec3 point = edge1.lerp(t);
            intersectionPoints.push_back(point);
        }
    }

    // Check each edge of tri2 against tri1
    for (int i = 0; i < 3; ++i)
    {
        Segment edge2(tri2[i], tri2[(i + 1) % 3]);
        float t;

        if (Intersects(edge2, tri1, &t))
        {
            vec3 point = edge2.lerp(t);
            intersectionPoints.push_back(point);
        }
    }

    if (intersectionPoints.size() == 2)
    {
        if (rpoints)
        {
            *rpoints = std::make_pair(intersectionPoints[0], intersectionPoints[1]);;
        }
        return 2;
    }

    return 0;
}

////////////////////////////////////////////////////////////////////////
// Geometric relationships
////////////////////////////////////////////////////////////////////////

// Compute angle between two geometric entities (in radians;  use acos)
float AngleBetween(const Plane& plane1, const Plane& plane2)
{
     return acos(dot(plane1.normal(), plane2.normal()) / (plane1.normal().length() * plane2.normal().length()));
}

// Compute angle between two geometric entities (in radians;  use acos)
float AngleBetween(const Line& line1, const Line& line2)
{
    return acos(dot(line1.vector, line2.vector) / (line1.vector.length() * line2.vector.length()));
}

// Compute angle between two geometric entities (in radians;  use acos)
float AngleBetween(const Line& line, const Plane& plane)
{
    return M_PI_2 - acos(dot(plane.normal(), line.vector) / (plane.normal().length() * line.vector.length()));
}

// Determine if two lines are coplanar
bool Coplanar(const Line& line1, const Line& line2)
{
    return abs(dot(line2.point - line1.point, cross(line1.vector, line2.vector))) < FLT_EPSILON;
}

bool Perpendicular(const vec3& v1, const vec3& v2)
{
    return abs(dot(v1, v2)) < FLT_EPSILON;
}

// Determine if two vectors are parallel.
bool Parallel(const vec3& v1, const vec3& v2)
{
    return abs(dot(cross(v1, v2), cross(v1, v2))) < FLT_EPSILON;
}

// Determine if two geometric entities are parallel.
bool Parallel(const Line& line1, const Line& line2)
{
    return Parallel(line1.vector, line2.vector);
}

// Determine if two geometric entities are parallel.
bool Parallel(const Line& line, const Plane& plane)
{
    return Perpendicular(line.vector, plane.normal());
}

// Determine if two geometric entities are parallel.
bool Parallel(const Plane& plane1, const Plane& plane2)
{
    return Parallel(plane1.normal(), plane2.normal());
}

// Determine if two geometric entities are perpendicular.
bool Perpendicular(const Line& line1, const Line& line2)
{
    return Perpendicular(line1.vector, line2.vector);
}

// Determine if two geometric entities are perpendicular.
bool Perpendicular(const Line& line, const Plane& plane)
{
    return Parallel(line.vector, plane.normal());
}

// Determine if two geometric entities are perpendicular.
bool Perpendicular(const Plane& plane1, const Plane& plane2)
{
    return Perpendicular(plane1.normal(), plane2.normal());
}
