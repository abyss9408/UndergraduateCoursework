
///////////////////////////////////////////////////////////////////////
// Unit tests of the geometry library.
////////////////////////////////////////////////////////////////////////

#include <cassert>
#include <iostream>
#include <fstream>

#include "geomlib.h"

using namespace std;

const float tolerance = 5e-3f;

int failures = 0;
int completed = 0;
int missing = 0;
int expected = 128;
#define AssertHere _AssertHere(__FILE__,__LINE__)
#define AssertNotHere(msg) _AssertNotHere(msg, __FILE__,__LINE__)
#define AssertEq(A,B,msg) _AssertEq(msg, A, B, __FILE__, __LINE__)

bool _AssertHere(const char* file, int line)
{
    completed++;
    //std::cout << "Passed "  << file << ":" << line << std::endl;
    return true;
}

bool _AssertNotHere(const char* name, const char* file, int line)
{
    completed++;
    if (!failures++) std::cout << "Failed tests:" <<  std::endl;
    std::cout << "  " << name << "  At line "  << file << ":" << line << std::endl;
    return false;
}

bool _AssertEq(const char* name, const float A, const float B,
               const char* file, int line)
{
    completed++;
    if (fabs(A-B) > tolerance) {
        if (!failures++) std::cout << "Failed tests:" <<  std::endl;
        std::cout << "  " << name << "  At line " << file << ":" << line << std::endl;
        return false; }
    else {
        //std::cout << "Passed " << file << ":" << line << std::endl;
        return true; }
}

bool _AssertEq(const char* name, const vec3 A, const vec3 B,
               const char* file, int line)
{
    completed++;
    if (fabs(A[0]-B[0]) > tolerance ||
        fabs(A[1]-B[1]) > tolerance ||
        fabs(A[2]-B[2]) > tolerance)
    {
        std::cout << "  " << name << "  At line " << file << ":" << line << std::endl;
        failures++;
        return false; }
    else {
        //std::cout << "Passed " << file << ":" << line << std::endl;
        return true; }
}

bool _AssertEq(const char* name, const vec4 A, const vec4 B,
               const char* file, int line)
{
    completed++;
    if (fabs(A[0]-B[0]) > tolerance ||
        fabs(A[1]-B[1]) > tolerance ||
        fabs(A[2]-B[2]) > tolerance ||
        fabs(A[3]-B[3]) > tolerance) {
        std::cout << "  " << name << "  At line " << file << ":" << line << std::endl;
        failures++;
        return false; }
    else {
        //std::cout << "Passed " << file << ":" << line << std::endl;
        return true; }
}

bool _AssertEq(const char* name, const mat4 A, const mat4 B,
               const char* file, int line)
{
    completed++;
    return A==B;
}

void PointLineDistance()
{

    vec3  P(9.0f, 6.0f, 5.0f);
    vec3 V(-83.0f, 52.0f, 344.65f);
    Line line(P, V);
    vec3 N = normalize(cross(V, vec3(83.0f, 52.0f, 344.65f)));

    AssertEq(Distance(P,line), 0.0,
             "Distance of P to line<P,V>.");
    AssertEq(Distance(P+V,line), 0.0,
             "Distance of P+V to line<P,V>");

    for (float d=1;     d<=2;  d+=0.5) {
        vec3 Q0 = P + d*N;
        AssertEq(Distance(Q0,line), d,
                 "Distance of P+d*N to line<P,V>");
        vec3 Q1 = P + V - d*N;
        AssertEq(Distance(Q1,line), d,
                 "Distance of P+d*N to line<P,V>"); }
}

void PointPlaneDistance()
{
    vec3 P(121.0f, -1290.2f, 347.04f);
    const float a=3.1f, b=4.2f, c=5.3f;
    const float d=6.1f, e=7.2f, f=8.3f;
    vec3 U(a,b,c);
    vec3 V(d,e,f);
    vec3 N = normalize(cross(U,V));
    Plane plane(N[0], N[1], N[2], -dot(vec3(P[0],P[1],P[2]),N));

    AssertEq(Distance(P,plane), 0.0,
             "Distance of P to plane containing P");
    AssertEq(Distance(P+U+V,plane), 0.0,
             "Distance of P+U+V to plane containing P");

    for (float d=1;     d<=2;  d+=0.5) {
        vec3 Q0 = P + U + V + d*N;
        AssertEq(Distance(Q0,plane), d,
                 "Distance of point from plane.");
        vec3 Q1 = P + U - V - d*N;
        AssertEq(Distance(Q1,plane), d,
                 "Distance of point from plane."); }
}

void AngleBetweenPlanes()
{
    const float a=3.1f, b=4.2f, c=5.3f;
    const float d=6.1f, e=7.2f, f=8.3f;
    Plane P(a, b, c, 12.0);
    Plane Q(d, e, f, 13.0);

    float cos0 = cos(AngleBetween(P,Q));
    float cos1 = dot(normalize(P.normal()), normalize(Q.normal()));

    AssertEq(cos0, cos1, "Angle between planes.");
}

void AngleBetweenLinePlane()
{
    vec3 V(-37.0f, 31.0f, -23.0f);
    vec3 W(44.0f, -345.0f, 883.0f);
    Plane P(V[0], V[1], V[2], 1.0f);
    Line L(vec3(255.45f, -42.78f, 363.0f), W);

    float cos0 = cos(AngleBetween(L,P));
    float cos1 = dot(V,W)/(length(V)*length(W));

    AssertEq(cos0*cos0 + cos1*cos1, 1.0f, "Angle between line and plane.");
}

void AngleBetweenLines()
{
    vec3 V(-37.0f, 31.0f, -23.0f);
    vec3 W(44.0f, -345.0f, 883.0f);
    Line M(vec3(323.0f, 45.0f, -457.0f), V);
    Line L(vec3(255.45f, -42.78f, 363.0f), W);

    float cos0 = cos(AngleBetween(M,L));
    float cos1 = dot(V,W);
    float l = length(V)*length(W);

    AssertEq(cos0, cos1/l, "Angle between lines.");
}

void CoplanarLines()
{
    vec3 Base(1.0f,2.0f,3.0f);
    vec3 N(0.1f, 0.2f, 1.0f);
    vec3 V (1.0f, 0.3f, 0.2f);
    vec3 W=cross(N,V);

    if (Coplanar(Line(Base,V),Line(Base,W))) {
        AssertHere; }
    else {
        AssertNotHere("Coplanar lines with same base."); }

    if (Coplanar(Line(Base,V),Line(Base+V,W))) {
        AssertHere; }
    else {
        AssertNotHere("Coplanar lines with different base."); }

    if (Coplanar(Line(Base,V),Line(Base+N,W))) {
        AssertNotHere("NON-Coplanar lines with different base."); }
    else {
        AssertHere; }
}

void ParallelPerpendicular()
{
    vec3 P0(1,1,0);
    vec3 P1(1,0,1);
    vec3 V(1,2,3);
    vec3 W = cross(V, vec3(1,0,1));

    Line LW0 = Line(P0,W);
    Line LW1 = Line(P1,W);
    Line LV1 = Line(P1,V);

    Plane PW0(W[0], W[1], W[2], 0);
    Plane PW1(W[0], W[1], W[2], 1);
    Plane PV1(V[0], V[1], V[2], 1);

    AssertEq(true,  Parallel(LW0, LW1), "Parallel lines.");
    AssertEq(false, Parallel(LW0, LV1), "Not parallel lines.");
    AssertEq(false, Perpendicular(LW0, LW1), "Not perpendicular lines.");
    AssertEq(true,  Perpendicular(LW0, LV1), "Perpendicular lines.");

    AssertEq(true,  Parallel(PW0, PW1), "Parallel planes.");
    AssertEq(false, Parallel(PW0, PV1), "Not parallel planes.");
    AssertEq(false, Perpendicular(PW0, PW1), "Not perpendicular planes.");
    AssertEq(true,  Perpendicular(PW0, PV1), "Perpendicular planes.");

    AssertEq(false,  Parallel(LW0, PW1), "Not parallel line/plane.");
    AssertEq(true, Parallel(LW0, PV1), "Parallel line/plane.");
    AssertEq(true, Perpendicular(LW0, PW1), "Perpendicular line/plane.");
    AssertEq(false,  Perpendicular(LW0, PV1), "Not perpendicular line/plane.");
       
}

void LinePlaneIntersection()
{
    Line first(vec3(2.23f, 123.0f, 401.0f), vec3(123.0f, -13.0f, 63.3f));
    Plane second(32.0f, 3.5f, 21.0f, -12.0f);

    float t;
    if (Intersects(first, second, &t)) {
        AssertEq(first.lerp(t),
                 vec3(-2.0774658530978e+02f,
                         1.4519264722786e+02f,
                         2.9293887926740e+02f),
                 "Line/Plane intersection point."); }
    else {
        AssertNotHere("Line/Plane intersection."); }

    try {
        Intersects(first, second);
        AssertHere; }
    catch(...) {
        AssertNotHere("Line/Plane intersection with NULL pointer."); }

    first.point = vec3(0.0f, 0.0f, 1.0f);
    first.vector = vec3(0.0f, 0.0f, 1.0f);
    second[0] = 1.0f;
    second[1] = 0.0f;
    second[2] = 0.0f;
    second[3] = 0.0f;

    if (Intersects(first, second)) {
        AssertNotHere("Line/Plane non-intersection."); }
    else {
        AssertHere; }
}

void SegmentPlaneIntersection()
{
    //vec3 base(2.23f, 123.0f, 401.0f);
    //vec3 vec(-123.0f, -13.0f, 63.3f);
    vec3 base(0,0,0);
    vec3 vec(1,1,1);
    
    Segment seg(base, base+vec);
    //Plane plane(3.2f, 3.5f, .21f, -120.0f);
    Plane plane(1, 1, 1, -1);

    float t;
    if (Intersects(seg, plane, &t))
    {
        vec3 point = seg.lerp(t);
        AssertEq(0.0, Distance(point,plane),
                 "Segment/Plane intersection point on plane.");
        AssertEq(0.0, Distance(point,Line(base,vec)),
                 "Segment/Plane intersection point on line.");
        AssertEq(true, seg.contains(point),
                 "Segment/Plane intersection point on Segment.");
    }
    else {
        AssertNotHere("Segment/Plane intersection."); }

    try {
        Intersects(seg, plane);
        AssertHere; }
    catch(...) {
        AssertNotHere("Segment/Plane intersection with NULL pointer."); }

    seg.point1 = vec3(0, 0, 1);
    seg.point2 = seg.point1+vec3(0, 0, 1);
    plane[0] = 1;
    plane[1] = 0;
    plane[2] = 0;
    plane[3] = 0;

    if (Intersects(seg, plane)) {
        AssertNotHere("Segment/Plane non-intersection."); }
    else {
        AssertHere; }
}

void RayPointContainment()
{ float t;

    {   Ray RAY(vec3(), vec3(1.0f, 1.0f, 1.0f));
        vec3 P(-2.0f, -2.0f, -2.0f);
        if (RAY.contains(P, &t)) {
            AssertNotHere("Ray/Point non-containment"); }
        else {
            AssertHere; }
    }

    {   Ray RAY(vec3(1.0f, 0.0f, 2.0f), vec3(4.0f, 8.0f, 6.0f));
        vec3 P = RAY.origin + 0.5f*RAY.direction;

        if (RAY.contains(P, &t)) {
            AssertEq(t, 0.5f, "Ray/Point containment parameter value."); }
        else {
            AssertNotHere("Ray/Point containment."); }

        try {
            RAY.contains(P);
            AssertHere; }
        catch(...) {
            AssertNotHere("Ray/Point containment with NULL pointer."); }
    }

}

void BoxPointContainment()
{
    Box BOX(vec3(0.5f, 0.5f, 0.5f), vec3(0.5f, 0.5f, 0.5f));
    
    vec3 points[] = {
        vec3(),
        vec3(1.0f,1.0f,1.0f),
        vec3(0.1f, 0.1f, 0.1f)};

    for (int i=0;  i<sizeof(points)/sizeof(points[0]);  i++) {
        if (BOX.contains(points[i])) {
            AssertHere; }
        else {
            AssertNotHere("Box/Point containment."); }
        
        if (BOX.contains(points[i]+vec3(2.0f,0.0f,0.0f))) {
            AssertNotHere("Box/Point non-containment"); }
        else {
            AssertHere; }
        
        if (BOX.contains(points[i]+vec3(-2.0f,0.0f,0.0f))) {
            AssertNotHere("Box/Point non-containment"); }
        else {
            AssertHere; }
        
        if (BOX.contains(points[i]+vec3(0.0f,2.0f,0.0f))) {
            AssertNotHere("Box/Point non-containment"); }
        else {
            AssertHere; }
        
        if (BOX.contains(points[i]+vec3(0.0f,-2.0f,0.0f))) {
            AssertNotHere("Box/Point non-containment"); }
        else {
            AssertHere; } }
}

void SegmentPointContainment()
{
    {   Segment seg(vec3(1.0f, 0.0f, 2.0f), vec3(5.0f, 8.0f, 8.0f));
        vec3 P(3.0f, 4.0f, 5.0f);

        if (seg.contains(P)) {
            AssertHere; }
        else {
            AssertNotHere("Segment/Point containment."); }
    }

    {   Segment seg(vec3(), vec3(1.0f, 1.0f, 1.0f));
        vec3 P(2.0f, 2.0f, 2.0f);
        if (seg.contains(P)) {
            AssertNotHere("Segment/Point non-containment"); }
        else {
            AssertHere; }
    }

    {   Segment seg(vec3(), vec3(0.0f, 0.0f, 1.0f));
        vec3 P(0.0f, 0.0f, 2.0f);
        if (seg.contains(P)) {
            AssertNotHere("Z-parallel-Segment/Point non-containment"); }
        else {
            AssertHere; }
    }
        
    {   Segment seg(vec3(), vec3(0.0f, 0.0f, 1.0f));
        vec3 P(0.0f, 0.0f, 0.5f);
        if (seg.contains(P)) {
            AssertHere; }
        else {
            AssertNotHere("Z-parallel-Segment/Point containment"); }
    }
}

void TrianglePointContainment()
{
    vec3 A(4.0f, -2.0f, 4.0f);
    vec3 B(0.0f, -2.0f, 2.0f);
    vec3 C(3.0f, -2.0f, -1.0f);
    Triangle TRI(A,B,C);
    vec3 In = A + 0.4f*(B-A) + 0.3f*(C-A);
    vec3 Out0 = A + 0.6f*(B-A) + 0.7f*(C-A);
    vec3 Out1 = A - 0.4f*(B-A) + 0.3f*(C-A);
    vec3 Out2 = A + 0.4f*(B-A) - 0.3f*(C-A);

    if (TRI.contains(In)) {
        AssertHere; }
    else {
        AssertNotHere("Triangle/Point containment."); }

    if (TRI.contains(Out0)) {
        AssertNotHere("Triangle/Point non-containment."); }
    else {
        AssertHere; }

    if (TRI.contains(Out1)) {
        AssertNotHere("Triangle/Point non-containment."); }
    else {
        AssertHere; }

    if (TRI.contains(Out2)) {
        AssertNotHere("Triangle/Point non-containment."); }
    else {
        AssertHere; }
}

void SegmentTriangleIntersection()
{
    Segment first(vec3(3, 0, 0), vec3(3, 3, 6));
    Triangle second(vec3(0, 0, 6), vec3(6, 0, 6),
                      vec3(3, 3, 0));
    float t;
    vec3 point;
    if (Intersects(first, second, &t)) {
        AssertEq(first.lerp(t), vec3(3,1.5,3),
                 "Segment/Triangle intersection point."); }
    else {
        AssertNotHere("Segment/Triangle intersection."); }

    first.point1 = vec3(10, 10, 10);
    first.point2 = vec3(20, 20, 20);
    second[0] = vec3(1, 0, 0);
    second[1] = vec3(-1, 1, 0);
    second[2] = vec3(0, -1, 0);

    if (Intersects(first, second)) {
        AssertNotHere("Segment/Triangle non-intersection"); }
    else {
        AssertHere; }
}

void TriangleIntersection()
{
    Triangle first(vec3(0, 0, 6), vec3(6, 0, 6),
                     vec3(3, 3, 0));
    Triangle second(vec3(0, 3, 4), vec3(6, 3, 4),
                      vec3(3.0f, 0.0f, 3.0f));

    std::pair <vec3, vec3> points;
    int intersection_count = Intersects(first, second, &points);

    if (intersection_count == 2)
    {
        if (points.first[0] > points.second[0]) {
            AssertEq(points.first,  vec3(4.285714f, 1.285714f, 3.428571f),
                     "Triangle/Triangle intersection -- first point.");
            AssertEq(points.second, vec3(1.714285f, 1.285714f, 3.428571f),
                     "Triangle/Triangle intersection -- second point."); }
        else {
            AssertEq(points.first,  vec3(1.714285f, 1.285714f, 3.428571f),
                     "Triangle/Triangle intersection -- first point.");
            AssertEq(points.second, vec3(4.285714f, 1.285714f, 3.428571f),
                     "Triangle/Triangle intersection -- second point."); } }
    else {
        AssertNotHere("Triangle/Triangle intersection."); }

    try {
        Intersects(first, second);
        AssertHere; }
    catch(...) {
        AssertNotHere("Triangle/Triangle intersection with NULL pointer.");}
}

void RaySphereIntersection()
{
    Ray first(vec3(0, 4, 6), vec3(5, 3, -5));
    Sphere second(vec3(2.5, 5.5, 3.5), 1);
    
    float t;
    int i = Intersects(first, second, &t);

    if (i) {
        AssertEq(length(first.lerp(t)-second.center),  second.radius,
                 "Ray/Sphere intersection."); }
    else {
        AssertNotHere("Ray/Sphere intersection."); }

    try {
        Intersects(first, second);
        AssertHere; }
    catch(...) {
        AssertNotHere("Ray/Sphere intersection with NULL pointer."); }
}

void TestRayBox(vec3 M, vec3 V, int n, vec3 A, vec3 B)
{
    Ray ray(M,V);
    Box box(vec3(0.5, 0.5, 0.5), vec3(0.5, 0.5, 0.5));
    float t;
    int i = Intersects(ray, box, &t);

    if (n==0)
        AssertEq(i, false, "Ray/Box non-intersection.");
    else {
        AssertEq(i, true, "Ray/Box intersection.");
        AssertEq(ray.lerp(t), A, "Ray/Box intersection."); }
}

void RayBoxIntersection()
{   
    TestRayBox(vec3( 0.50, 0.50, 0.50), vec3(-1.00, 0.00, 0.00),
               1, vec3( 0.00, 0.50, 0.50), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 0.50), vec3(-1.00, 0.00, 0.00),
               1, vec3( 0.00, 0.50, 0.50), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 0.50), vec3(-1.00, 0.00, 0.00),
               1, vec3( 0.00, 0.50, 0.50), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3(-0.50, 0.50, 0.50), vec3( 1.00, 0.00, 0.00),
               2, vec3( 0.00, 0.50, 0.50), vec3( 1.00, 0.50, 0.50));
    TestRayBox(vec3(-0.50, 0.50, 0.50), vec3( 1.00, 0.00, 0.00),
               2, vec3( 0.00, 0.50, 0.50), vec3( 1.00, 0.50, 0.50));
    TestRayBox(vec3(-0.50, 0.50, 0.50), vec3( 1.00, 0.00, 0.00),
               2, vec3( 0.00, 0.50, 0.50), vec3( 1.00, 0.50, 0.50));
    TestRayBox(vec3(-0.50, 0.50, 0.50), vec3( 1.00, 0.00, 0.00),
               2, vec3( 0.00, 0.50, 0.50), vec3( 1.00, 0.50, 0.50));
    TestRayBox(vec3(-0.50, 0.50, 0.50), vec3( 1.00, 0.00, 0.00),
               2, vec3( 0.00, 0.50, 0.50), vec3( 1.00, 0.50, 0.50));
    TestRayBox(vec3(-0.50, 0.50, 0.50), vec3( 1.00, 0.00, 0.00),
               2, vec3( 0.00, 0.50, 0.50), vec3( 1.00, 0.50, 0.50));
    TestRayBox(vec3(-0.50, 1.00, 0.50), vec3( 1.00, 0.00, 0.00),
               2, vec3( 0.00, 1.00, 0.50), vec3( 1.00, 1.00, 0.50));
    TestRayBox(vec3(-0.50, 1.00, 0.50), vec3( 1.00, 0.00, 0.00),
               2, vec3( 0.00, 1.00, 0.50), vec3( 1.00, 1.00, 0.50));
    TestRayBox(vec3(-0.50, 1.00, 0.50), vec3( 1.00, 0.00, 0.00),
               2, vec3( 0.00, 1.00, 0.50), vec3( 1.00, 1.00, 0.50));
    TestRayBox(vec3(-0.50, 1.50, 0.50), vec3( 1.00, 0.00, 0.00),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3(-0.50, 1.50, 0.50), vec3( 1.00, 0.00, 0.00),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3(-0.50, 1.50, 0.50), vec3( 1.00, 0.00, 0.00),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 0.50), vec3(-1.00, 0.00, 0.00),
               1, vec3( 0.00, 0.50, 0.50), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 0.50), vec3(-1.00, 0.50, 0.00),
               1, vec3( 0.00, 0.75, 0.50), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 0.50), vec3(-1.00, 0.00, 0.50),
               1, vec3( 0.00, 0.50, 0.75), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3(-0.50, 0.50, 0.50), vec3( 1.00, 0.00, 0.00),
               2, vec3( 0.00, 0.50, 0.50), vec3( 1.00, 0.50, 0.50));
    TestRayBox(vec3(-0.50, 0.50, 0.50), vec3( 1.00, 0.50, 0.00),
               2, vec3( 0.00, 0.75, 0.50), vec3( 0.50, 1.00, 0.50));
    TestRayBox(vec3(-0.50, 0.50, 0.50), vec3( 1.00, 0.00, 0.50),
               2, vec3( 0.00, 0.50, 0.75), vec3( 0.50, 0.50, 1.00));
    TestRayBox(vec3(-0.50, 0.50, 0.50), vec3( 1.00, 0.00, 0.00),
               2, vec3( 0.00, 0.50, 0.50), vec3( 1.00, 0.50, 0.50));
    TestRayBox(vec3(-0.50, 0.50, 0.50), vec3( 1.00, 0.50, 0.00),
               2, vec3( 0.00, 0.75, 0.50), vec3( 0.50, 1.00, 0.50));
    TestRayBox(vec3(-0.50, 0.50, 0.50), vec3( 1.00, 0.00, 0.50),
               2, vec3( 0.00, 0.50, 0.75), vec3( 0.50, 0.50, 1.00));
    TestRayBox(vec3(-0.50, 1.00, 0.50), vec3( 1.00, 0.00, 0.00),
               2, vec3( 0.00, 1.00, 0.50), vec3( 1.00, 1.00, 0.50));
    TestRayBox(vec3(-0.50, 1.00, 0.50), vec3( 1.00, 0.50, 0.00),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3(-0.50, 1.00, 0.50), vec3( 1.00, 0.00, 0.50),
               2, vec3( 0.00, 1.00, 0.75), vec3( 0.50, 1.00, 1.00));
    TestRayBox(vec3(-0.50, 1.50, 0.50), vec3( 1.00, 0.00, 0.00),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3(-0.50, 1.50, 0.50), vec3( 1.00, 0.50, 0.00),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3(-0.50, 1.50, 0.50), vec3( 1.00, 0.00, 0.50),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 0.50), vec3( 1.00, 0.00, 0.00),
               1, vec3( 1.00, 0.50, 0.50), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 0.50), vec3( 1.00, 0.00, 0.00),
               1, vec3( 1.00, 0.50, 0.50), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 0.50), vec3( 1.00, 0.00, 0.00),
               1, vec3( 1.00, 0.50, 0.50), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 1.50, 0.50, 0.50), vec3(-1.00, 0.00, 0.00),
               2, vec3( 1.00, 0.50, 0.50), vec3( 0.00, 0.50, 0.50));
    TestRayBox(vec3( 1.50, 0.50, 0.50), vec3(-1.00, 0.00, 0.00),
               2, vec3( 1.00, 0.50, 0.50), vec3( 0.00, 0.50, 0.50));
    TestRayBox(vec3( 1.50, 0.50, 0.50), vec3(-1.00, 0.00, 0.00),
               2, vec3( 1.00, 0.50, 0.50), vec3( 0.00, 0.50, 0.50));
    TestRayBox(vec3( 1.50, 0.50, 0.50), vec3(-1.00, 0.00, 0.00),
               2, vec3( 1.00, 0.50, 0.50), vec3( 0.00, 0.50, 0.50));
    TestRayBox(vec3( 1.50, 0.50, 0.50), vec3(-1.00, 0.00, 0.00),
               2, vec3( 1.00, 0.50, 0.50), vec3( 0.00, 0.50, 0.50));
    TestRayBox(vec3( 1.50, 0.50, 0.50), vec3(-1.00, 0.00, 0.00),
               2, vec3( 1.00, 0.50, 0.50), vec3( 0.00, 0.50, 0.50));
    TestRayBox(vec3( 1.50, 1.00, 0.50), vec3(-1.00, 0.00, 0.00),
               2, vec3( 1.00, 1.00, 0.50), vec3( 0.00, 1.00, 0.50));
    TestRayBox(vec3( 1.50, 1.00, 0.50), vec3(-1.00, 0.00, 0.00),
               2, vec3( 1.00, 1.00, 0.50), vec3( 0.00, 1.00, 0.50));
    TestRayBox(vec3( 1.50, 1.00, 0.50), vec3(-1.00, 0.00, 0.00),
               2, vec3( 1.00, 1.00, 0.50), vec3( 0.00, 1.00, 0.50));
    TestRayBox(vec3( 1.50, 1.50, 0.50), vec3(-1.00, 0.00, 0.00),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 1.50, 1.50, 0.50), vec3(-1.00, 0.00, 0.00),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 1.50, 1.50, 0.50), vec3(-1.00, 0.00, 0.00),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 0.50), vec3( 1.00, 0.00, 0.00),
               1, vec3( 1.00, 0.50, 0.50), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 0.50), vec3( 1.00, 0.50, 0.00),
               1, vec3( 1.00, 0.75, 0.50), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 0.50), vec3( 1.00, 0.00, 0.50),
               1, vec3( 1.00, 0.50, 0.75), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 1.50, 0.50, 0.50), vec3(-1.00, 0.00, 0.00),
               2, vec3( 1.00, 0.50, 0.50), vec3( 0.00, 0.50, 0.50));
    TestRayBox(vec3( 1.50, 0.50, 0.50), vec3(-1.00, 0.50, 0.00),
               2, vec3( 1.00, 0.75, 0.50), vec3( 0.50, 1.00, 0.50));
    TestRayBox(vec3( 1.50, 0.50, 0.50), vec3(-1.00, 0.00, 0.50),
               2, vec3( 1.00, 0.50, 0.75), vec3( 0.50, 0.50, 1.00));
    TestRayBox(vec3( 1.50, 0.50, 0.50), vec3(-1.00, 0.00, 0.00),
               2, vec3( 1.00, 0.50, 0.50), vec3( 0.00, 0.50, 0.50));
    TestRayBox(vec3( 1.50, 0.50, 0.50), vec3(-1.00, 0.50, 0.00),
               2, vec3( 1.00, 0.75, 0.50), vec3( 0.50, 1.00, 0.50));
    TestRayBox(vec3( 1.50, 0.50, 0.50), vec3(-1.00, 0.00, 0.50),
               2, vec3( 1.00, 0.50, 0.75), vec3( 0.50, 0.50, 1.00));
    TestRayBox(vec3( 1.50, 1.00, 0.50), vec3(-1.00, 0.00, 0.00),
               2, vec3( 1.00, 1.00, 0.50), vec3( 0.00, 1.00, 0.50));
    TestRayBox(vec3( 1.50, 1.00, 0.50), vec3(-1.00, 0.50, 0.00),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 1.50, 1.00, 0.50), vec3(-1.00, 0.00, 0.50),
               2, vec3( 1.00, 1.00, 0.75), vec3( 0.50, 1.00, 1.00));
    TestRayBox(vec3( 1.50, 1.50, 0.50), vec3(-1.00, 0.00, 0.00),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 1.50, 1.50, 0.50), vec3(-1.00, 0.50, 0.00),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 1.50, 1.50, 0.50), vec3(-1.00, 0.00, 0.50),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 0.50), vec3( 0.00, 0.00,-1.00),
               1, vec3( 0.50, 0.50, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 0.50), vec3( 0.00, 0.00,-1.00),
               1, vec3( 0.50, 0.50, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 0.50), vec3( 0.00, 0.00,-1.00),
               1, vec3( 0.50, 0.50, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 0.50,-0.50), vec3( 0.00, 0.00, 1.00),
               2, vec3( 0.50, 0.50, 0.00), vec3( 0.50, 0.50, 1.00));
    TestRayBox(vec3( 0.50, 0.50,-0.50), vec3( 0.00, 0.00, 1.00),
               2, vec3( 0.50, 0.50, 0.00), vec3( 0.50, 0.50, 1.00));
    TestRayBox(vec3( 0.50, 0.50,-0.50), vec3( 0.00, 0.00, 1.00),
               2, vec3( 0.50, 0.50, 0.00), vec3( 0.50, 0.50, 1.00));
    TestRayBox(vec3( 0.50, 0.50,-0.50), vec3( 0.00, 0.00, 1.00),
               2, vec3( 0.50, 0.50, 0.00), vec3( 0.50, 0.50, 1.00));
    TestRayBox(vec3( 0.50, 0.50,-0.50), vec3( 0.00, 0.00, 1.00),
               2, vec3( 0.50, 0.50, 0.00), vec3( 0.50, 0.50, 1.00));
    TestRayBox(vec3( 0.50, 0.50,-0.50), vec3( 0.00, 0.00, 1.00),
               2, vec3( 0.50, 0.50, 0.00), vec3( 0.50, 0.50, 1.00));
    TestRayBox(vec3( 1.00, 0.50,-0.50), vec3( 0.00, 0.00, 1.00),
               2, vec3( 1.00, 0.50, 0.00), vec3( 1.00, 0.50, 1.00));
    TestRayBox(vec3( 1.00, 0.50,-0.50), vec3( 0.00, 0.00, 1.00),
               2, vec3( 1.00, 0.50, 0.00), vec3( 1.00, 0.50, 1.00));
    TestRayBox(vec3( 1.00, 0.50,-0.50), vec3( 0.00, 0.00, 1.00),
               2, vec3( 1.00, 0.50, 0.00), vec3( 1.00, 0.50, 1.00));
    TestRayBox(vec3( 1.50, 0.50,-0.50), vec3( 0.00, 0.00, 1.00),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 1.50, 0.50,-0.50), vec3( 0.00, 0.00, 1.00),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 1.50, 0.50,-0.50), vec3( 0.00, 0.00, 1.00),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 0.50), vec3( 0.00, 0.00,-1.00),
               1, vec3( 0.50, 0.50, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 0.50), vec3( 0.50, 0.00,-1.00),
               1, vec3( 0.75, 0.50, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 0.50), vec3( 0.00, 0.50,-1.00),
               1, vec3( 0.50, 0.75, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 0.50,-0.50), vec3( 0.00, 0.00, 1.00),
               2, vec3( 0.50, 0.50, 0.00), vec3( 0.50, 0.50, 1.00));
    TestRayBox(vec3( 0.50, 0.50,-0.50), vec3( 0.50, 0.00, 1.00),
               2, vec3( 0.75, 0.50, 0.00), vec3( 1.00, 0.50, 0.50));
    TestRayBox(vec3( 0.50, 0.50,-0.50), vec3( 0.00, 0.50, 1.00),
               2, vec3( 0.50, 0.75, 0.00), vec3( 0.50, 1.00, 0.50));
    TestRayBox(vec3( 0.50, 0.50,-0.50), vec3( 0.00, 0.00, 1.00),
               2, vec3( 0.50, 0.50, 0.00), vec3( 0.50, 0.50, 1.00));
    TestRayBox(vec3( 0.50, 0.50,-0.50), vec3( 0.50, 0.00, 1.00),
               2, vec3( 0.75, 0.50, 0.00), vec3( 1.00, 0.50, 0.50));
    TestRayBox(vec3( 0.50, 0.50,-0.50), vec3( 0.00, 0.50, 1.00),
               2, vec3( 0.50, 0.75, 0.00), vec3( 0.50, 1.00, 0.50));
    TestRayBox(vec3( 1.00, 0.50,-0.50), vec3( 0.00, 0.00, 1.00),
               2, vec3( 1.00, 0.50, 0.00), vec3( 1.00, 0.50, 1.00));
    TestRayBox(vec3( 1.00, 0.50,-0.50), vec3( 0.50, 0.00, 1.00),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 1.00, 0.50,-0.50), vec3( 0.00, 0.50, 1.00),
               2, vec3( 1.00, 0.75, 0.00), vec3( 1.00, 1.00, 0.50));
    TestRayBox(vec3( 1.50, 0.50,-0.50), vec3( 0.00, 0.00, 1.00),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 1.50, 0.50,-0.50), vec3( 0.50, 0.00, 1.00),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 1.50, 0.50,-0.50), vec3( 0.00, 0.50, 1.00),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 0.50), vec3( 0.00, 0.00, 1.00),
               1, vec3( 0.50, 0.50, 1.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 0.50), vec3( 0.00, 0.00, 1.00),
               1, vec3( 0.50, 0.50, 1.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 0.50), vec3( 0.00, 0.00, 1.00),
               1, vec3( 0.50, 0.50, 1.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 1.50), vec3( 0.00, 0.00,-1.00),
               2, vec3( 0.50, 0.50, 1.00), vec3( 0.50, 0.50, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 1.50), vec3( 0.00, 0.00,-1.00),
               2, vec3( 0.50, 0.50, 1.00), vec3( 0.50, 0.50, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 1.50), vec3( 0.00, 0.00,-1.00),
               2, vec3( 0.50, 0.50, 1.00), vec3( 0.50, 0.50, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 1.50), vec3( 0.00, 0.00,-1.00),
               2, vec3( 0.50, 0.50, 1.00), vec3( 0.50, 0.50, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 1.50), vec3( 0.00, 0.00,-1.00),
               2, vec3( 0.50, 0.50, 1.00), vec3( 0.50, 0.50, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 1.50), vec3( 0.00, 0.00,-1.00),
               2, vec3( 0.50, 0.50, 1.00), vec3( 0.50, 0.50, 0.00));
    TestRayBox(vec3( 1.00, 0.50, 1.50), vec3( 0.00, 0.00,-1.00),
               2, vec3( 1.00, 0.50, 1.00), vec3( 1.00, 0.50, 0.00));
    TestRayBox(vec3( 1.00, 0.50, 1.50), vec3( 0.00, 0.00,-1.00),
               2, vec3( 1.00, 0.50, 1.00), vec3( 1.00, 0.50, 0.00));
    TestRayBox(vec3( 1.00, 0.50, 1.50), vec3( 0.00, 0.00,-1.00),
               2, vec3( 1.00, 0.50, 1.00), vec3( 1.00, 0.50, 0.00));
    TestRayBox(vec3( 1.50, 0.50, 1.50), vec3( 0.00, 0.00,-1.00),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 1.50, 0.50, 1.50), vec3( 0.00, 0.00,-1.00),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 1.50, 0.50, 1.50), vec3( 0.00, 0.00,-1.00),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 0.50), vec3( 0.00, 0.00, 1.00),
               1, vec3( 0.50, 0.50, 1.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 0.50), vec3( 0.50, 0.00, 1.00),
               1, vec3( 0.75, 0.50, 1.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 0.50), vec3( 0.00, 0.50, 1.00),
               1, vec3( 0.50, 0.75, 1.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 1.50), vec3( 0.00, 0.00,-1.00),
               2, vec3( 0.50, 0.50, 1.00), vec3( 0.50, 0.50, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 1.50), vec3( 0.50, 0.00,-1.00),
               2, vec3( 0.75, 0.50, 1.00), vec3( 1.00, 0.50, 0.50));
    TestRayBox(vec3( 0.50, 0.50, 1.50), vec3( 0.00, 0.50,-1.00),
               2, vec3( 0.50, 0.75, 1.00), vec3( 0.50, 1.00, 0.50));
    TestRayBox(vec3( 0.50, 0.50, 1.50), vec3( 0.00, 0.00,-1.00),
               2, vec3( 0.50, 0.50, 1.00), vec3( 0.50, 0.50, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 1.50), vec3( 0.50, 0.00,-1.00),
               2, vec3( 0.75, 0.50, 1.00), vec3( 1.00, 0.50, 0.50));
    TestRayBox(vec3( 0.50, 0.50, 1.50), vec3( 0.00, 0.50,-1.00),
               2, vec3( 0.50, 0.75, 1.00), vec3( 0.50, 1.00, 0.50));
    TestRayBox(vec3( 1.00, 0.50, 1.50), vec3( 0.00, 0.00,-1.00),
               2, vec3( 1.00, 0.50, 1.00), vec3( 1.00, 0.50, 0.00));
    TestRayBox(vec3( 1.00, 0.50, 1.50), vec3( 0.50, 0.00,-1.00),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 1.00, 0.50, 1.50), vec3( 0.00, 0.50,-1.00),
               2, vec3( 1.00, 0.75, 1.00), vec3( 1.00, 1.00, 0.50));
    TestRayBox(vec3( 1.50, 0.50, 1.50), vec3( 0.00, 0.00,-1.00),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 1.50, 0.50, 1.50), vec3( 0.50, 0.00,-1.00),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 1.50, 0.50, 1.50), vec3( 0.00, 0.50,-1.00),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 0.50), vec3( 0.00,-1.00, 0.00),
               1, vec3( 0.50, 0.00, 0.50), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 0.50), vec3( 0.00,-1.00, 0.00),
               1, vec3( 0.50, 0.00, 0.50), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 0.50), vec3( 0.00,-1.00, 0.00),
               1, vec3( 0.50, 0.00, 0.50), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50,-0.50, 0.50), vec3( 0.00, 1.00, 0.00),
               2, vec3( 0.50, 0.00, 0.50), vec3( 0.50, 1.00, 0.50));
    TestRayBox(vec3( 0.50,-0.50, 0.50), vec3( 0.00, 1.00, 0.00),
               2, vec3( 0.50, 0.00, 0.50), vec3( 0.50, 1.00, 0.50));
    TestRayBox(vec3( 0.50,-0.50, 0.50), vec3( 0.00, 1.00, 0.00),
               2, vec3( 0.50, 0.00, 0.50), vec3( 0.50, 1.00, 0.50));
    TestRayBox(vec3( 0.50,-0.50, 0.50), vec3( 0.00, 1.00, 0.00),
               2, vec3( 0.50, 0.00, 0.50), vec3( 0.50, 1.00, 0.50));
    TestRayBox(vec3( 0.50,-0.50, 0.50), vec3( 0.00, 1.00, 0.00),
               2, vec3( 0.50, 0.00, 0.50), vec3( 0.50, 1.00, 0.50));
    TestRayBox(vec3( 0.50,-0.50, 0.50), vec3( 0.00, 1.00, 0.00),
               2, vec3( 0.50, 0.00, 0.50), vec3( 0.50, 1.00, 0.50));
    TestRayBox(vec3( 0.50,-0.50, 1.00), vec3( 0.00, 1.00, 0.00),
               2, vec3( 0.50, 0.00, 1.00), vec3( 0.50, 1.00, 1.00));
    TestRayBox(vec3( 0.50,-0.50, 1.00), vec3( 0.00, 1.00, 0.00),
               2, vec3( 0.50, 0.00, 1.00), vec3( 0.50, 1.00, 1.00));
    TestRayBox(vec3( 0.50,-0.50, 1.00), vec3( 0.00, 1.00, 0.00),
               2, vec3( 0.50, 0.00, 1.00), vec3( 0.50, 1.00, 1.00));
    TestRayBox(vec3( 0.50,-0.50, 1.50), vec3( 0.00, 1.00, 0.00),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50,-0.50, 1.50), vec3( 0.00, 1.00, 0.00),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50,-0.50, 1.50), vec3( 0.00, 1.00, 0.00),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 0.50), vec3( 0.00,-1.00, 0.00),
               1, vec3( 0.50, 0.00, 0.50), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 0.50), vec3( 0.00,-1.00, 0.50),
               1, vec3( 0.50, 0.00, 0.75), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 0.50), vec3( 0.50,-1.00, 0.00),
               1, vec3( 0.75, 0.00, 0.50), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50,-0.50, 0.50), vec3( 0.00, 1.00, 0.00),
               2, vec3( 0.50, 0.00, 0.50), vec3( 0.50, 1.00, 0.50));
    TestRayBox(vec3( 0.50,-0.50, 0.50), vec3( 0.00, 1.00, 0.50),
               2, vec3( 0.50, 0.00, 0.75), vec3( 0.50, 0.50, 1.00));
    TestRayBox(vec3( 0.50,-0.50, 0.50), vec3( 0.50, 1.00, 0.00),
               2, vec3( 0.75, 0.00, 0.50), vec3( 1.00, 0.50, 0.50));
    TestRayBox(vec3( 0.50,-0.50, 0.50), vec3( 0.00, 1.00, 0.00),
               2, vec3( 0.50, 0.00, 0.50), vec3( 0.50, 1.00, 0.50));
    TestRayBox(vec3( 0.50,-0.50, 0.50), vec3( 0.00, 1.00, 0.50),
               2, vec3( 0.50, 0.00, 0.75), vec3( 0.50, 0.50, 1.00));
    TestRayBox(vec3( 0.50,-0.50, 0.50), vec3( 0.50, 1.00, 0.00),
               2, vec3( 0.75, 0.00, 0.50), vec3( 1.00, 0.50, 0.50));
    TestRayBox(vec3( 0.50,-0.50, 1.00), vec3( 0.00, 1.00, 0.00),
               2, vec3( 0.50, 0.00, 1.00), vec3( 0.50, 1.00, 1.00));
    TestRayBox(vec3( 0.50,-0.50, 1.00), vec3( 0.00, 1.00, 0.50),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50,-0.50, 1.00), vec3( 0.50, 1.00, 0.00),
               2, vec3( 0.75, 0.00, 1.00), vec3( 1.00, 0.50, 1.00));
    TestRayBox(vec3( 0.50,-0.50, 1.50), vec3( 0.00, 1.00, 0.00),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50,-0.50, 1.50), vec3( 0.00, 1.00, 0.50),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50,-0.50, 1.50), vec3( 0.50, 1.00, 0.00),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 0.50), vec3( 0.00, 1.00, 0.00),
               1, vec3( 0.50, 1.00, 0.50), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 0.50), vec3( 0.00, 1.00, 0.00),
               1, vec3( 0.50, 1.00, 0.50), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 0.50), vec3( 0.00, 1.00, 0.00),
               1, vec3( 0.50, 1.00, 0.50), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 1.50, 0.50), vec3( 0.00,-1.00, 0.00),
               2, vec3( 0.50, 1.00, 0.50), vec3( 0.50, 0.00, 0.50));
    TestRayBox(vec3( 0.50, 1.50, 0.50), vec3( 0.00,-1.00, 0.00),
               2, vec3( 0.50, 1.00, 0.50), vec3( 0.50, 0.00, 0.50));
    TestRayBox(vec3( 0.50, 1.50, 0.50), vec3( 0.00,-1.00, 0.00),
               2, vec3( 0.50, 1.00, 0.50), vec3( 0.50, 0.00, 0.50));
    TestRayBox(vec3( 0.50, 1.50, 0.50), vec3( 0.00,-1.00, 0.00),
               2, vec3( 0.50, 1.00, 0.50), vec3( 0.50, 0.00, 0.50));
    TestRayBox(vec3( 0.50, 1.50, 0.50), vec3( 0.00,-1.00, 0.00),
               2, vec3( 0.50, 1.00, 0.50), vec3( 0.50, 0.00, 0.50));
    TestRayBox(vec3( 0.50, 1.50, 0.50), vec3( 0.00,-1.00, 0.00),
               2, vec3( 0.50, 1.00, 0.50), vec3( 0.50, 0.00, 0.50));
    TestRayBox(vec3( 0.50, 1.50, 1.00), vec3( 0.00,-1.00, 0.00),
               2, vec3( 0.50, 1.00, 1.00), vec3( 0.50, 0.00, 1.00));
    TestRayBox(vec3( 0.50, 1.50, 1.00), vec3( 0.00,-1.00, 0.00),
               2, vec3( 0.50, 1.00, 1.00), vec3( 0.50, 0.00, 1.00));
    TestRayBox(vec3( 0.50, 1.50, 1.00), vec3( 0.00,-1.00, 0.00),
               2, vec3( 0.50, 1.00, 1.00), vec3( 0.50, 0.00, 1.00));
    TestRayBox(vec3( 0.50, 1.50, 1.50), vec3( 0.00,-1.00, 0.00),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 1.50, 1.50), vec3( 0.00,-1.00, 0.00),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 1.50, 1.50), vec3( 0.00,-1.00, 0.00),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 0.50), vec3( 0.00, 1.00, 0.00),
               1, vec3( 0.50, 1.00, 0.50), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 0.50), vec3( 0.00, 1.00, 0.50),
               1, vec3( 0.50, 1.00, 0.75), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 0.50, 0.50), vec3( 0.50, 1.00, 0.00),
               1, vec3( 0.75, 1.00, 0.50), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 1.50, 0.50), vec3( 0.00,-1.00, 0.00),
               2, vec3( 0.50, 1.00, 0.50), vec3( 0.50, 0.00, 0.50));
    TestRayBox(vec3( 0.50, 1.50, 0.50), vec3( 0.00,-1.00, 0.50),
               2, vec3( 0.50, 1.00, 0.75), vec3( 0.50, 0.50, 1.00));
    TestRayBox(vec3( 0.50, 1.50, 0.50), vec3( 0.50,-1.00, 0.00),
               2, vec3( 0.75, 1.00, 0.50), vec3( 1.00, 0.50, 0.50));
    TestRayBox(vec3( 0.50, 1.50, 0.50), vec3( 0.00,-1.00, 0.00),
               2, vec3( 0.50, 1.00, 0.50), vec3( 0.50, 0.00, 0.50));
    TestRayBox(vec3( 0.50, 1.50, 0.50), vec3( 0.00,-1.00, 0.50),
               2, vec3( 0.50, 1.00, 0.75), vec3( 0.50, 0.50, 1.00));
    TestRayBox(vec3( 0.50, 1.50, 0.50), vec3( 0.50,-1.00, 0.00),
               2, vec3( 0.75, 1.00, 0.50), vec3( 1.00, 0.50, 0.50));
    TestRayBox(vec3( 0.50, 1.50, 1.00), vec3( 0.00,-1.00, 0.00),
               2, vec3( 0.50, 1.00, 1.00), vec3( 0.50, 0.00, 1.00));
    TestRayBox(vec3( 0.50, 1.50, 1.00), vec3( 0.00,-1.00, 0.50),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 1.50, 1.00), vec3( 0.50,-1.00, 0.00),
               2, vec3( 0.75, 1.00, 1.00), vec3( 1.00, 0.50, 1.00));
    TestRayBox(vec3( 0.50, 1.50, 1.50), vec3( 0.00,-1.00, 0.00),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 1.50, 1.50), vec3( 0.00,-1.00, 0.50),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
    TestRayBox(vec3( 0.50, 1.50, 1.50), vec3( 0.50,-1.00, 0.00),
               0, vec3( 0.00, 0.00, 0.00), vec3( 0.00, 0.00, 0.00));
}



void TestRayTriangle(bool exp_ret, float exp_t, float exp_u, float exp_v,
                     const vec3& M, const vec3& V,
                     const vec3& A, const vec3& B, const vec3& C)
{
    Ray ray(M,V);
    Triangle tri(A,B,C);
    float t, u, v;
    
    bool ret = Intersects(ray, tri, &t, &u, &v);

    if (exp_ret) {
        AssertEq(ret, true, "Ray/Triangle intersection returns false incorrectly.");
        AssertEq(t, exp_t, "Ray/Triangle intersection returns wrong t value.");
        AssertEq(u, exp_u, "Ray/Triangle intersection returns wrong u value.");
        AssertEq(v, exp_v, "Ray/Triangle intersection returns wrong v value."); }
    else
        AssertEq(ret, false, "Ray/Triangle intersection returns true incorrectly.");
}

void RayTriangleIntersection()
{
    TestRayTriangle(1, 10.087703, 0.831203, 0.004474, vec3(-0.000000, -5.000000, 1.510000), vec3(0.247272, 0.434101, -0.020327), vec3(2.552866, -0.514804, 1.855846), vec3(2.480001, -0.641009, 1.196617), vec3(3.023869, -0.786740, 1.196617));
    TestRayTriangle(1, 10.101251, 0.836096, 0.106659, vec3(0.000000, -5.000000, 1.510000), vec3(0.244742, 0.435532, -0.020327), vec3(2.208069, -0.170004, 1.855846), vec3(2.480001, -0.641009, 1.196617), vec3(2.552866, -0.514804, 1.855846));
    TestRayTriangle(1, 10.103123, 0.777766, 0.058387, vec3(0.000000, -5.000000, 1.510000), vec3(0.249794, 0.432654, -0.020327), vec3(2.552866, -0.514804, 1.855846), vec3(2.480001, -0.641009, 1.196617), vec3(3.023869, -0.786740, 1.196617));
    TestRayTriangle(1, 10.117785, 0.836605, 0.043520, vec3(0.000000, -5.000000, 1.510000), vec3(0.242205, 0.436948, -0.020327), vec3(2.208069, -0.170004, 1.855846), vec3(2.480001, -0.641009, 1.196617), vec3(2.552866, -0.514804, 1.855846));
    TestRayTriangle(1, 10.118937, 0.724174, 0.112466, vec3(-0.000000, -5.000000, 1.510000), vec3(0.252306, 0.431193, -0.020327), vec3(2.552866, -0.514804, 1.855846), vec3(2.480001, -0.641009, 1.196617), vec3(3.023869, -0.786740, 1.196617));
    TestRayTriangle(1, 10.134710, 0.017185, 0.819941, vec3(0.000000, -5.000000, 1.510000), vec3(0.239658, 0.438350, -0.020327), vec3(2.208069, -0.170004, 1.855846), vec3(2.081863, -0.242869, 1.196617), vec3(2.480001, -0.641009, 1.196617));
    TestRayTriangle(1, 10.135123, 0.670429, 0.166711, vec3(-0.000000, -5.000000, 1.510000), vec3(0.254811, 0.429718, -0.020327), vec3(2.552866, -0.514804, 1.855846), vec3(2.480001, -0.641009, 1.196617), vec3(3.023869, -0.786740, 1.196617));
    TestRayTriangle(1, 10.135535, 0.085976, 0.107286, vec3(0.000000, -5.000000, 1.510000), vec3(0.043518, 0.497416, -0.026139), vec3(0.457490, 0.000000, 1.307019), vec3(0.343118, 0.198099, 0.586454), vec3(0.396198, 0.228745, 1.307019));
    TestRayTriangle(1, 10.152037, 0.072252, 0.765409, vec3(0.000000, -5.000000, 1.510000), vec3(0.237104, 0.439736, -0.020327), vec3(2.208069, -0.170004, 1.855846), vec3(2.081863, -0.242869, 1.196617), vec3(2.480001, -0.641009, 1.196617));
    TestRayTriangle(1, 10.168694, 0.562436, 0.275740, vec3(0.000000, -5.000000, 1.510000), vec3(0.259793, 0.426724, -0.020327), vec3(2.552866, -0.514804, 1.855846), vec3(2.480001, -0.641009, 1.196617), vec3(3.023869, -0.786740, 1.196617));
    TestRayTriangle(1, 10.186068, 0.508185, 0.330525, vec3(0.000000, -5.000000, 1.510000), vec3(0.262271, 0.425206, -0.020327), vec3(2.552866, -0.514804, 1.855846), vec3(2.480001, -0.641009, 1.196617), vec3(3.023869, -0.786740, 1.196617));
    TestRayTriangle(1, 10.187891, 0.182991, 0.655777, vec3(-0.000000, -5.000000, 1.510000), vec3(0.231972, 0.442465, -0.020327), vec3(2.208069, -0.170004, 1.855846), vec3(2.081863, -0.242869, 1.196617), vec3(2.480001, -0.641009, 1.196617));
    TestRayTriangle(1, 10.206421, 0.238673, 0.600667, vec3(0.000000, -4.999999, 1.510000), vec3(0.229394, 0.443808, -0.020327), vec3(2.208069, -0.170004, 1.855846), vec3(2.081863, -0.242869, 1.196617), vec3(2.480001, -0.641009, 1.196617));
    TestRayTriangle(1, 10.222049, 0.399142, 0.440678, vec3(0.000000, -5.000000, 1.510000), vec3(0.267201, 0.422125, -0.020327), vec3(2.552866, -0.514804, 1.855846), vec3(2.480001, -0.641009, 1.196617), vec3(3.023869, -0.786740, 1.196617));
    TestRayTriangle(1, 10.225369, 0.294566, 0.545357, vec3(0.000000, -5.000000, 1.510000), vec3(0.226808, 0.445135, -0.020327), vec3(2.208069, -0.170004, 1.855846), vec3(2.081863, -0.242869, 1.196617), vec3(2.480001, -0.641009, 1.196617));
    TestRayTriangle(1, 10.244731, 0.350682, 0.489839, vec3(-0.000000, -5.000000, 1.510000), vec3(0.224214, 0.446447, -0.020327), vec3(2.208069, -0.170004, 1.855846), vec3(2.081863, -0.242869, 1.196617), vec3(2.480001, -0.641009, 1.196617));
    TestRayTriangle(1, 10.259650, 0.289362, 0.551617, vec3(0.000000, -5.000000, 1.510000), vec3(0.272095, 0.418988, -0.020327), vec3(2.552866, -0.514804, 1.855846), vec3(2.480001, -0.641009, 1.196617), vec3(3.023869, -0.786740, 1.196617));
    TestRayTriangle(1, 10.279079, 0.234179, 0.607400, vec3(0.000000, -5.000000, 1.510000), vec3(0.274527, 0.417398, -0.020327), vec3(2.552866, -0.514804, 1.855846), vec3(2.480001, -0.641009, 1.196617), vec3(3.023869, -0.786740, 1.196617));
    TestRayTriangle(1, 10.284719, 0.463581, 0.378172, vec3(0.000000, -5.000000, 1.510000), vec3(0.219004, 0.449025, -0.020327), vec3(2.208069, -0.170004, 1.855846), vec3(2.081863, -0.242869, 1.196617), vec3(2.480001, -0.641009, 1.196617));
    TestRayTriangle(1, 10.298268, 0.091879, 0.467075, vec3(0.000000, -5.000000, 1.510000), vec3(0.040624, 0.497661, -0.026139), vec3(0.457490, 0.000000, 1.307019), vec3(0.343118, 0.198099, 0.586454), vec3(0.396198, 0.228745, 1.307019));
    TestRayTriangle(1, 10.319199, 0.123205, 0.719610, vec3(0.000000, -5.000000, 1.510000), vec3(0.279365, 0.414176, -0.020327), vec3(2.552866, -0.514804, 1.855846), vec3(2.480001, -0.641009, 1.196617), vec3(3.023869, -0.786740, 1.196617));
    TestRayTriangle(1, 10.326400, 0.577423, 0.265615, vec3(-0.000000, -5.000000, 1.510000), vec3(0.213765, 0.451543, -0.020327), vec3(2.208069, -0.170004, 1.855846), vec3(2.081863, -0.242869, 1.196617), vec3(2.480001, -0.641009, 1.196617));
    TestRayTriangle(1, 10.339894, 0.067408, 0.776046, vec3(0.000000, -5.000000, 1.510000), vec3(0.281770, 0.412543, -0.020327), vec3(2.552866, -0.514804, 1.855846), vec3(2.480001, -0.641009, 1.196617), vec3(3.023869, -0.786740, 1.196617));
    TestRayTriangle(1, 10.347899, 0.634704, 0.208997, vec3(0.000000, -5.000000, 1.510000), vec3(0.211135, 0.452779, -0.020327), vec3(2.208069, -0.170004, 1.855846), vec3(2.081863, -0.242869, 1.196617), vec3(2.480001, -0.641009, 1.196617));
    TestRayTriangle(1, 10.361029, 0.011391, 0.832716, vec3(0.000000, -5.000000, 1.510000), vec3(0.284166, 0.410897, -0.020327), vec3(2.552866, -0.514804, 1.855846), vec3(2.480001, -0.641009, 1.196617), vec3(3.023869, -0.786740, 1.196617));
    TestRayTriangle(1, 10.369830, 0.692238, 0.152140, vec3(0.000000, -5.000000, 1.510000), vec3(0.208497, 0.454000, -0.020327), vec3(2.208069, -0.170004, 1.855846), vec3(2.081863, -0.242869, 1.196617), vec3(2.480001, -0.641009, 1.196617));
    TestRayTriangle(1, 10.404594, 0.845449, 0.116975, vec3(0.000000, -5.000000, 1.510000), vec3(0.288927, 0.407563, -0.020327), vec3(2.552866, -0.514804, 1.855846), vec3(3.023869, -0.786740, 1.196617), vec3(3.023869, -0.641011, 1.855846));
    TestRayTriangle(1, 10.415023, 0.808077, 0.037694, vec3(0.000000, -5.000000, 1.510000), vec3(0.203200, 0.456395, -0.020327), vec3(2.208069, -0.170004, 1.855846), vec3(2.081863, -0.242869, 1.196617), vec3(2.480001, -0.641009, 1.196617));
    TestRayTriangle(1, 10.448724, 0.810450, 0.036360, vec3(-0.000000, -5.000000, 1.510000), vec3(0.291293, 0.405876, -0.020327), vec3(3.023869, -0.641011, 1.855846), vec3(3.023869, -0.786740, 1.196617), vec3(3.567737, -0.641013, 1.196617));
    TestRayTriangle(1, 10.455997, 0.847034, 0.118850, vec3(0.000000, -5.000000, 1.510000), vec3(0.200542, 0.457570, -0.020327), vec3(2.081865, 0.301000, 1.855846), vec3(2.081863, -0.242869, 1.196617), vec3(2.208069, -0.170004, 1.855846));
    TestRayTriangle(1, 10.466672, 0.097988, 0.838594, vec3(-0.000000, -5.000000, 1.510000), vec3(0.037728, 0.497889, -0.026139), vec3(0.457490, 0.000000, 1.307019), vec3(0.343118, 0.198099, 0.586454), vec3(0.396198, 0.228745, 1.307019));
    TestRayTriangle(1, 10.531693, 0.849368, 0.016708, vec3(0.000000, -5.000000, 1.510000), vec3(0.197876, 0.458728, -0.020327), vec3(2.081865, 0.301000, 1.855846), vec3(2.081863, -0.242869, 1.196617), vec3(2.208069, -0.170004, 1.855846));
    TestRayTriangle(1, 10.538947, 0.100610, 0.054883, vec3(-0.000000, -5.000000, 1.510000), vec3(0.034831, 0.498099, -0.026139), vec3(0.396198, 0.228745, 1.307019), vec3(0.198100, 0.343118, 0.586454), vec3(0.228745, 0.396198, 1.307019));
    TestRayTriangle(1, 10.592385, 0.102550, 0.224814, vec3(-0.000000, -5.000001, 1.510000), vec3(0.031932, 0.498294, -0.026139), vec3(0.396198, 0.228745, 1.307019), vec3(0.198100, 0.343118, 0.586454), vec3(0.228745, 0.396198, 1.307019));
    TestRayTriangle(1, 10.597896, 0.643515, 0.207895, vec3(0.000000, -5.000000, 1.510000), vec3(0.295996, 0.402458, -0.020327), vec3(3.023869, -0.641011, 1.855846), vec3(3.023869, -0.786740, 1.196617), vec3(3.567737, -0.641013, 1.196617));
    TestRayTriangle(1, 10.608853, 0.075286, 0.776461, vec3(0.000000, -5.000000, 1.510000), vec3(0.195204, 0.459871, -0.020327), vec3(2.081865, 0.301000, 1.855846), vec3(1.936135, 0.301000, 1.196617), vec3(2.081863, -0.242869, 1.196617));
    TestRayTriangle(1, 10.640360, 0.190054, 0.297636, vec3(-0.000000, -5.000000, 1.510000), vec3(0.029013, 0.498134, -0.031947), vec3(0.396198, 0.228745, 1.307019), vec3(0.198100, 0.343118, 0.586454), vec3(0.228745, 0.396198, 1.307019));
    TestRayTriangle(1, 10.643362, 0.147298, 0.347040, vec3(0.000000, -5.000000, 1.510000), vec3(0.029023, 0.498311, -0.029043), vec3(0.396198, 0.228745, 1.307019), vec3(0.198100, 0.343118, 0.586454), vec3(0.228745, 0.396198, 1.307019));
    TestRayTriangle(1, 10.646732, 0.104520, 0.396471, vec3(-0.000000, -5.000001, 1.510000), vec3(0.029033, 0.498471, -0.026139), vec3(0.396198, 0.228745, 1.307019), vec3(0.198100, 0.343118, 0.586454), vec3(0.228745, 0.396198, 1.307019));
    TestRayTriangle(1, 10.674608, 0.558274, 0.295501, vec3(0.000000, -5.000000, 1.510000), vec3(0.298333, 0.400730, -0.020327), vec3(3.023869, -0.641011, 1.855846), vec3(3.023869, -0.786740, 1.196617), vec3(3.567737, -0.641013, 1.196617));
    TestRayTriangle(1, 10.685689, 0.407077, 0.221812, vec3(0.000000, -5.000000, 1.510000), vec3(0.026055, 0.497155, -0.046446), vec3(0.396198, 0.228745, 1.307019), vec3(0.198100, 0.343118, 0.586454), vec3(0.228745, 0.396198, 1.307019));
    TestRayTriangle(1, 10.686936, 0.364190, 0.271478, vec3(0.000000, -5.000000, 1.510000), vec3(0.026069, 0.497416, -0.043549), vec3(0.396198, 0.228745, 1.307019), vec3(0.198100, 0.343118, 0.586454), vec3(0.228745, 0.396198, 1.307019));
    TestRayTriangle(1, 10.687509, 0.166380, 0.687792, vec3(0.000000, -5.000000, 1.510000), vec3(0.192525, 0.460999, -0.020327), vec3(2.081865, 0.301000, 1.855846), vec3(1.936135, 0.301000, 1.196617), vec3(2.081863, -0.242869, 1.196617));
    TestRayTriangle(1, 10.690495, 0.278378, 0.370861, vec3(0.000000, -5.000000, 1.510000), vec3(0.026093, 0.497890, -0.037750), vec3(0.396198, 0.228745, 1.307019), vec3(0.198100, 0.343118, 0.586454), vec3(0.228745, 0.396198, 1.307019));
    TestRayTriangle(1, 10.692827, 0.235450, 0.420576, vec3(0.000000, -5.000000, 1.510000), vec3(0.026104, 0.498100, -0.034849), vec3(0.396198, 0.228745, 1.307019), vec3(0.198100, 0.343118, 0.586454), vec3(0.228745, 0.396198, 1.307019));
    TestRayTriangle(1, 10.739882, 0.539965, 0.244349, vec3(0.000000, -5.000000, 1.510000), vec3(0.023121, 0.496413, -0.055127), vec3(0.396198, 0.228745, 1.307019), vec3(0.198100, 0.343118, 0.586454), vec3(0.228745, 0.396198, 1.307019));
    TestRayTriangle(1, 10.740077, 0.496876, 0.294365, vec3(0.000000, -5.000000, 1.510000), vec3(0.023135, 0.496725, -0.052235), vec3(0.396198, 0.228745, 1.307019), vec3(0.198100, 0.343118, 0.586454), vec3(0.228745, 0.396198, 1.307019));
    TestRayTriangle(1, 10.752809, 0.471796, 0.384390, vec3(0.000000, -5.000000, 1.510000), vec3(0.300659, 0.398987, -0.020327), vec3(3.023869, -0.641011, 1.855846), vec3(3.023869, -0.786740, 1.196617), vec3(3.567737, -0.641013, 1.196617));
    TestRayTriangle(1, 10.767698, 0.258848, 0.597797, vec3(0.000000, -5.000000, 1.510000), vec3(0.189840, 0.462111, -0.020327), vec3(2.081865, 0.301000, 1.855846), vec3(1.936135, 0.301000, 1.196617), vec3(2.081863, -0.242869, 1.196617));
    TestRayTriangle(1, 10.797223, 0.630940, 0.316820, vec3(0.000000, -5.000000, 1.510000), vec3(0.020205, 0.495864, -0.060906), vec3(0.396198, 0.228745, 1.307019), vec3(0.198100, 0.343118, 0.586454), vec3(0.228745, 0.396198, 1.307019));
    TestRayTriangle(1, 10.813566, 0.443229, 0.462500, vec3(0.000000, -5.000000, 1.510000), vec3(0.302898, 0.397130, -0.023234), vec3(3.023869, -0.641011, 1.855846), vec3(3.023869, -0.786740, 1.196617), vec3(3.567737, -0.641013, 1.196617));
    TestRayTriangle(1, 10.826633, 0.572669, 0.104122, vec3(0.000000, -5.000000, 1.510000), vec3(0.017307, 0.495611, -0.063792), vec3(0.228745, 0.396198, 1.307019), vec3(0.198100, 0.343118, 0.586454), vec3(0.000001, 0.396198, 0.586454));
    TestRayTriangle(1, 10.832521, 0.384059, 0.474586, vec3(-0.000000, -5.000000, 1.510000), vec3(0.302975, 0.397232, -0.020327), vec3(3.023869, -0.641011, 1.855846), vec3(3.023869, -0.786740, 1.196617), vec3(3.567737, -0.641013, 1.196617));
    TestRayTriangle(1, 10.832891, 0.672467, 0.091560, vec3(0.000000, -4.999999, 1.510000), vec3(0.017280, 0.494836, -0.069558), vec3(0.228745, 0.396198, 1.307019), vec3(0.198100, 0.343118, 0.586454), vec3(0.000001, 0.396198, 0.586454));
    TestRayTriangle(1, 10.849467, 0.352724, 0.506442, vec3(0.000000, -5.000000, 1.510000), vec3(0.187148, 0.463208, -0.020327), vec3(2.081865, 0.301000, 1.855846), vec3(1.936135, 0.301000, 1.196617), vec3(2.081863, -0.242869, 1.196617));
    TestRayTriangle(1, 10.849868, 0.562075, 0.127458, vec3(0.000000, -5.000000, 1.510000), vec3(0.005669, 0.487158, -0.112447), vec3(0.198100, 0.343118, 0.586454), vec3(0.000000, 0.228746, 0.058963), vec3(0.000001, 0.396198, 0.586454));
    TestRayTriangle(1, 10.851527, 0.736993, 0.108397, vec3(0.000000, -5.000000, 1.510000), vec3(0.002823, 0.485147, -0.120933), vec3(0.198100, 0.343118, 0.586454), vec3(0.000000, 0.228746, 0.058963), vec3(0.000001, 0.396198, 0.586454));
    TestRayTriangle(1, 10.854016, 0.969182, 0.000003, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482209, -0.132189), vec3(0.000001, 0.396198, 0.586454), vec3(0.000000, 0.228746, 0.058963), vec3(-0.114373, 0.198100, 0.058963));
    TestRayTriangle(1, 10.855088, 0.504825, 0.184147, vec3(-0.000000, -5.000000, 1.510000), vec3(0.005676, 0.487804, -0.109611), vec3(0.198100, 0.343118, 0.586454), vec3(0.000000, 0.228746, 0.058963), vec3(0.000001, 0.396198, 0.586454));
    TestRayTriangle(1, 10.856720, 0.912074, 0.000003, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482970, -0.129381), vec3(0.000001, 0.396198, 0.586454), vec3(0.000000, 0.228746, 0.058963), vec3(-0.114373, 0.198100, 0.058963));
    TestRayTriangle(1, 10.857120, 0.212256, 0.161810, vec3(0.000000, -5.000000, 1.510000), vec3(0.011421, 0.490685, -0.095376), vec3(0.198100, 0.343118, 0.586454), vec3(0.000000, 0.228746, 0.058963), vec3(0.000001, 0.396198, 0.586454));
    TestRayTriangle(1, 10.858027, 0.329839, 0.201746, vec3(0.000000, -5.000000, 1.510000), vec3(0.008546, 0.489601, -0.101080), vec3(0.198100, 0.343118, 0.586454), vec3(0.000000, 0.228746, 0.058963), vec3(0.000001, 0.396198, 0.586454));
    TestRayTriangle(1, 10.859618, 0.668085, 0.228594, vec3(0.000000, -5.000000, 1.510000), vec3(0.014363, 0.493639, -0.078188), vec3(0.228745, 0.396198, 1.307019), vec3(0.198100, 0.343118, 0.586454), vec3(0.000001, 0.396198, 0.586454));
    TestRayTriangle(1, 10.859787, 0.854933, 0.000003, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.483715, -0.126569), vec3(0.000001, 0.396198, 0.586454), vec3(0.000000, 0.228746, 0.058963), vec3(-0.114373, 0.198100, 0.058963));
    TestRayTriangle(1, 10.860189, 0.622593, 0.222236, vec3(0.000000, -5.000000, 1.510000), vec3(0.002830, 0.486521, -0.115280), vec3(0.198100, 0.343118, 0.586454), vec3(0.000000, 0.228746, 0.058963), vec3(0.000001, 0.396198, 0.586454));
    TestRayTriangle(1, 10.860686, 0.447514, 0.240896, vec3(-0.000000, -5.000000, 1.510000), vec3(0.005683, 0.488433, -0.106771), vec3(0.198100, 0.343118, 0.586454), vec3(0.000000, 0.228746, 0.058963), vec3(0.000001, 0.396198, 0.586454));
    TestRayTriangle(1, 10.872308, 0.097159, 0.274656, vec3(0.000000, -5.000000, 1.510000), vec3(0.011446, 0.491762, -0.089659), vec3(0.198100, 0.343118, 0.586454), vec3(0.000000, 0.228746, 0.058963), vec3(0.000001, 0.396198, 0.586454));
    TestRayTriangle(1, 10.873004, 0.332709, 0.354576, vec3(0.000000, -5.000000, 1.510000), vec3(0.005698, 0.489643, -0.101080), vec3(0.198100, 0.343118, 0.586454), vec3(0.000000, 0.228746, 0.058963), vec3(0.000001, 0.396198, 0.586454));
    TestRayTriangle(1, 10.884556, 0.614137, 0.371966, vec3(-0.000000, -5.000000, 1.510000), vec3(0.011469, 0.492772, -0.083929), vec3(0.228745, 0.396198, 1.307019), vec3(0.198100, 0.343118, 0.586454), vec3(0.000001, 0.396198, 0.586454));
    TestRayTriangle(1, 10.894528, 0.354817, 0.553767, vec3(-0.000000, -5.000000, 1.510000), vec3(0.305203, 0.395362, -0.023234), vec3(3.023869, -0.641011, 1.855846), vec3(3.023869, -0.786740, 1.196617), vec3(3.567737, -0.641013, 1.196617));
    TestRayTriangle(1, 10.932855, 0.448048, 0.413691, vec3(0.000000, -5.000000, 1.510000), vec3(0.184450, 0.464289, -0.020327), vec3(2.081865, 0.301000, 1.855846), vec3(1.936135, 0.301000, 1.196617), vec3(2.081863, -0.242869, 1.196617));
    TestRayTriangle(1, 10.977094, 0.265074, 0.646421, vec3(0.000000, -5.000000, 1.510000), vec3(0.307498, 0.393579, -0.023234), vec3(3.023869, -0.641011, 1.855846), vec3(3.023869, -0.786740, 1.196617), vec3(3.567737, -0.641013, 1.196617));
    TestRayTriangle(1, 100.030815, 0.387979, 0.064808, vec3(0.000000, -5.000000, 1.510000), vec3(0.165353, 0.471294, -0.023234), vec3(16.363636, 40.909088, -0.971854), vec3(16.363636, 43.636364, -0.595334), vec3(19.090912, 43.636364, -0.791281));
    TestRayTriangle(1, 100.972557, 0.534916, 0.369761, vec3(-0.000000, -5.000000, 1.510000), vec3(0.334108, 0.371064, -0.026139), vec3(32.727272, 30.000000, -0.395683), vec3(32.727272, 32.727272, -1.235349), vec3(35.454548, 32.727272, -1.165070));
    TestRayTriangle(1, 109.924019, 0.039191, 0.509602, vec3(-0.000000, -5.000000, 1.510000), vec3(0.336153, 0.368995, -0.029043), vec3(35.454548, 35.454548, -1.679924), vec3(38.181816, 38.181816, -2.166917), vec3(38.181816, 35.454548, -1.647641));
    TestRayTriangle(1, 11.017905, 0.544854, 0.319506, vec3(0.000000, -5.000000, 1.510000), vec3(0.181746, 0.465354, -0.020327), vec3(2.081865, 0.301000, 1.855846), vec3(1.936135, 0.301000, 1.196617), vec3(2.081863, -0.242869, 1.196617));
    TestRayTriangle(1, 11.061277, 0.173980, 0.740481, vec3(-0.000000, -5.000000, 1.510000), vec3(0.309783, 0.391784, -0.023234), vec3(3.023869, -0.641011, 1.855846), vec3(3.023869, -0.786740, 1.196617), vec3(3.567737, -0.641013, 1.196617));
    TestRayTriangle(1, 11.104660, 0.643190, 0.223845, vec3(0.000000, -5.000000, 1.510000), vec3(0.179036, 0.466404, -0.020327), vec3(2.081865, 0.301000, 1.855846), vec3(1.936135, 0.301000, 1.196617), vec3(2.081863, -0.242869, 1.196617));
    TestRayTriangle(1, 11.147174, 0.081473, 0.836015, vec3(0.000000, -5.000000, 1.510000), vec3(0.312057, 0.389975, -0.023234), vec3(3.023869, -0.641011, 1.855846), vec3(3.023869, -0.786740, 1.196617), vec3(3.567737, -0.641013, 1.196617));
    TestRayTriangle(1, 11.193172, 0.743099, 0.126666, vec3(-0.000000, -5.000000, 1.510000), vec3(0.176319, 0.467438, -0.020327), vec3(2.081865, 0.301000, 1.855846), vec3(1.936135, 0.301000, 1.196617), vec3(2.081863, -0.242869, 1.196617));
    TestRayTriangle(1, 11.234745, 0.920573, 0.014371, vec3(0.000000, -5.000000, 1.510000), vec3(0.314320, 0.388153, -0.023234), vec3(3.023869, -0.641011, 1.855846), vec3(3.567737, -0.641013, 1.196617), vec3(3.494873, -0.514807, 1.855846));
    TestRayTriangle(1, 11.283482, 0.844624, 0.027927, vec3(-0.000000, -5.000000, 1.510000), vec3(0.173597, 0.468456, -0.020327), vec3(2.081865, 0.301000, 1.855846), vec3(1.936135, 0.301000, 1.196617), vec3(2.081863, -0.242869, 1.196617));
    TestRayTriangle(1, 11.347385, 0.172998, -0.000000, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482209, -0.132189), vec3(0.000000, 0.000000, 0.010000), vec3(0.000000, 2.727270, 0.010000), vec3(2.727270, 2.727270, 0.010000));
    TestRayTriangle(1, 11.423034, 0.000127, 0.500000, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482209, -0.132189), vec3(-2000.000000, -2000.000000, 0.000000), vec3(-2000.000000, 2000.000000, 0.000000), vec3(2000.000000, 2000.000000, 0.000000));
    TestRayTriangle(1, 11.540864, 0.703319, 0.228039, vec3(-0.000000, -5.000000, 1.510000), vec3(0.316573, 0.386318, -0.023234), vec3(3.494873, -0.514807, 1.855846), vec3(3.567737, -0.641013, 1.196617), vec3(3.965878, -0.242876, 1.196617));
    TestRayTriangle(1, 11.593627, 0.219770, 0.780230, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482970, -0.129381), vec3(-2.727274, 0.000000, 0.010000), vec3(0.000000, 2.727270, 0.010000), vec3(0.000000, 0.000000, 0.010000));
    TestRayTriangle(1, 11.593628, 0.219770, -0.000000, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482970, -0.129381), vec3(0.000000, 0.000000, 0.010000), vec3(0.000000, 2.727270, 0.010000), vec3(2.727270, 2.727270, 0.010000));
    TestRayTriangle(1, 11.670918, 0.000159, 0.500000, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482970, -0.129381), vec3(-2000.000000, -2000.000000, 0.000000), vec3(-2000.000000, 2000.000000, 0.000000), vec3(2000.000000, 2000.000000, 0.000000));
    TestRayTriangle(1, 11.851208, 0.268621, -0.000000, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.483715, -0.126569), vec3(0.000000, 0.000000, 0.010000), vec3(0.000000, 2.727270, 0.010000), vec3(2.727270, 2.727270, 0.010000));
    TestRayTriangle(1, 11.930217, 0.000193, 0.500000, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.483715, -0.126569), vec3(-2000.000000, -2000.000000, 0.000000), vec3(-2000.000000, 2000.000000, 0.000000), vec3(2000.000000, 2000.000000, 0.000000));
    TestRayTriangle(1, 111.114265, 0.120304, 0.662422, vec3(-0.000000, -5.000000, 1.510000), vec3(0.338294, 0.367033, -0.029043), vec3(35.454548, 35.454548, -1.679924), vec3(38.181816, 38.181816, -2.166917), vec3(38.181816, 35.454548, -1.647641));
    TestRayTriangle(1, 111.956215, 0.513588, 0.274248, vec3(0.000000, -5.000000, 1.510000), vec3(0.165353, 0.471294, -0.023234), vec3(16.363636, 46.363636, -0.502785), vec3(19.090912, 49.090912, -1.436546), vec3(19.090912, 46.363636, -0.899450));
    TestRayTriangle(1, 112.479774, 0.013420, 0.127290, vec3(0.000000, -5.000000, 1.510000), vec3(0.342541, 0.363072, -0.029043), vec3(38.181816, 35.454548, -1.647641), vec3(38.181816, 38.181816, -2.166917), vec3(40.909088, 38.181816, -2.450350));
    TestRayTriangle(1, 112.551697, 0.097383, 0.597175, vec3(0.000000, -5.000000, 1.510000), vec3(0.159858, 0.473186, -0.023234), vec3(16.363636, 46.363636, -0.502785), vec3(16.363636, 49.090912, -0.960606), vec3(19.090912, 49.090912, -1.436546));
    TestRayTriangle(1, 112.726212, 0.090895, 0.154413, vec3(-0.000000, -5.000000, 1.510000), vec3(0.344648, 0.361073, -0.029043), vec3(38.181816, 35.454548, -1.647641), vec3(40.909088, 38.181816, -2.450350), vec3(40.909088, 35.454548, -1.928317));
    TestRayTriangle(1, 112.989639, 0.300066, 0.508682, vec3(-0.000000, -5.000000, 1.510000), vec3(0.157102, 0.474108, -0.023234), vec3(16.363636, 46.363636, -0.502785), vec3(16.363636, 49.090912, -0.960606), vec3(19.090912, 49.090912, -1.436546));
    TestRayTriangle(1, 113.798286, 0.024253, 0.300390, vec3(0.000000, -5.000000, 1.510000), vec3(0.151575, 0.475904, -0.023234), vec3(16.363636, 49.090912, -0.960606), vec3(19.090912, 51.818184, -2.213088), vec3(19.090912, 49.090912, -1.436546));
    TestRayTriangle(1, 113.991074, 0.094428, 0.125101, vec3(0.000000, -5.000000, 1.510000), vec3(0.148804, 0.476778, -0.023234), vec3(16.363636, 49.090912, -0.960606), vec3(19.090912, 51.818184, -2.213088), vec3(19.090912, 49.090912, -1.436546));
    TestRayTriangle(1, 114.274048, 0.212979, 0.002105, vec3(0.000000, -5.000000, 1.510000), vec3(0.143247, 0.478477, -0.023234), vec3(16.363636, 49.090912, -0.960606), vec3(16.363636, 51.818184, -1.813969), vec3(19.090912, 51.818184, -2.213088));
    TestRayTriangle(1, 114.492233, 0.321963, 0.457480, vec3(0.000000, -5.000000, 1.510000), vec3(0.137670, 0.480111, -0.023234), vec3(13.636365, 49.090912, -0.573869), vec3(16.363636, 51.818184, -1.813969), vec3(16.363636, 49.090912, -0.960606));
    TestRayTriangle(1, 9.309615, 0.393178, 0.601082, vec3(0.000000, -5.000000, 1.510000), vec3(0.023135, 0.496725, -0.052235), vec3(-0.000002, -0.457490, 1.307019), vec3(0.198098, -0.343119, 0.586454), vec3(0.228743, -0.396199, 1.307019));
    TestRayTriangle(1, 9.309758, 0.064728, 0.141313, vec3(0.000000, -5.000000, 1.510000), vec3(0.026093, 0.497890, -0.037750), vec3(0.228743, -0.396199, 1.307019), vec3(0.198098, -0.343119, 0.586454), vec3(0.343117, -0.198101, 0.586454));
    TestRayTriangle(1, 9.319879, 0.543397, 0.234565, vec3(0.000000, -5.000000, 1.510000), vec3(0.017307, 0.495611, -0.063792), vec3(-0.000002, -0.457490, 1.307019), vec3(0.198098, -0.343119, 0.586454), vec3(0.228743, -0.396199, 1.307019));
    TestRayTriangle(1, 9.320236, 0.431354, 0.568503, vec3(0.000000, -5.000000, 1.510000), vec3(0.023121, 0.496413, -0.055127), vec3(-0.000002, -0.457490, 1.307019), vec3(0.198098, -0.343119, 0.586454), vec3(0.228743, -0.396199, 1.307019));
    TestRayTriangle(1, 9.325212, 0.506518, 0.385041, vec3(0.000000, -5.000000, 1.510000), vec3(0.020205, 0.495864, -0.060906), vec3(-0.000002, -0.457490, 1.307019), vec3(0.198098, -0.343119, 0.586454), vec3(0.228743, -0.396199, 1.307019));
    TestRayTriangle(1, 9.329887, 0.056751, 0.212818, vec3(-0.000000, -5.000001, 1.510000), vec3(0.029033, 0.498471, -0.026139), vec3(0.228743, -0.396199, 1.307019), vec3(0.343117, -0.198101, 0.586454), vec3(0.396197, -0.228747, 1.307019));
    TestRayTriangle(1, 9.332158, 0.122444, 0.159868, vec3(0.000000, -5.000000, 1.510000), vec3(0.026069, 0.497416, -0.043549), vec3(0.228743, -0.396199, 1.307019), vec3(0.198098, -0.343119, 0.586454), vec3(0.343117, -0.198101, 0.586454));
    TestRayTriangle(1, 9.339666, 0.094751, 0.188038, vec3(0.000000, -5.000000, 1.510000), vec3(0.029023, 0.498311, -0.029043), vec3(0.228743, -0.396199, 1.307019), vec3(0.343117, -0.198101, 0.586454), vec3(0.396197, -0.228747, 1.307019));
    TestRayTriangle(1, 9.343863, 0.151409, 0.169177, vec3(0.000000, -5.000000, 1.510000), vec3(0.026055, 0.497155, -0.046446), vec3(0.228743, -0.396199, 1.307019), vec3(0.198098, -0.343119, 0.586454), vec3(0.343117, -0.198101, 0.586454));
    TestRayTriangle(1, 9.344096, 0.620310, 0.168682, vec3(0.000000, -4.999999, 1.510000), vec3(0.017280, 0.494836, -0.069558), vec3(-0.000002, -0.457490, 1.307019), vec3(0.198098, -0.343119, 0.586454), vec3(0.228743, -0.396199, 1.307019));
    TestRayTriangle(1, 9.349788, 0.132835, 0.163204, vec3(-0.000000, -5.000000, 1.510000), vec3(0.029013, 0.498134, -0.031947), vec3(0.228743, -0.396199, 1.307019), vec3(0.343117, -0.198101, 0.586454), vec3(0.396197, -0.228747, 1.307019));
    TestRayTriangle(1, 9.366321, 0.055516, 0.679126, vec3(0.000000, -5.000000, 1.510000), vec3(0.014363, 0.493639, -0.078188), vec3(-0.000002, -0.457490, 1.307019), vec3(-0.000002, -0.396198, 0.586454), vec3(0.198098, -0.343119, 0.586454));
    TestRayTriangle(1, 9.377592, 0.267632, 0.542944, vec3(-0.000000, -5.000000, 1.510000), vec3(0.011469, 0.492772, -0.083929), vec3(-0.000002, -0.457490, 1.307019), vec3(-0.000002, -0.396198, 0.586454), vec3(0.198098, -0.343119, 0.586454));
    TestRayTriangle(1, 9.391846, 0.058999, 0.384646, vec3(-0.000000, -5.000001, 1.510000), vec3(0.031932, 0.498294, -0.026139), vec3(0.228743, -0.396199, 1.307019), vec3(0.343117, -0.198101, 0.586454), vec3(0.396197, -0.228747, 1.307019));
    TestRayTriangle(1, 9.406655, 0.345246, 0.543511, vec3(0.000000, -5.000000, 1.510000), vec3(0.011446, 0.491762, -0.089659), vec3(-0.000002, -0.457490, 1.307019), vec3(-0.000002, -0.396198, 0.586454), vec3(0.198098, -0.343119, 0.586454));
    TestRayTriangle(1, 9.437176, 0.423353, 0.544081, vec3(0.000000, -5.000000, 1.510000), vec3(0.011421, 0.490685, -0.095376), vec3(-0.000002, -0.457490, 1.307019), vec3(-0.000002, -0.396198, 0.586454), vec3(0.198098, -0.343119, 0.586454));
    TestRayTriangle(1, 9.452538, 0.060512, 0.236934, vec3(0.000000, -5.000000, 1.510000), vec3(0.005698, 0.489643, -0.101080), vec3(-0.000002, -0.396198, 0.586454), vec3(0.114372, -0.198100, 0.058963), vec3(0.198098, -0.343119, 0.586454));
    TestRayTriangle(1, 9.454961, 0.061287, 0.558771, vec3(-0.000000, -5.000000, 1.510000), vec3(0.034831, 0.498099, -0.026139), vec3(0.228743, -0.396199, 1.307019), vec3(0.343117, -0.198101, 0.586454), vec3(0.396197, -0.228747, 1.307019));
    TestRayTriangle(1, 9.469249, 0.063715, 0.371726, vec3(0.000000, -5.000000, 1.510000), vec3(0.008546, 0.489601, -0.101080), vec3(-0.000002, -0.396198, 0.586454), vec3(0.114372, -0.198100, 0.058963), vec3(0.198098, -0.343119, 0.586454));
    TestRayTriangle(1, 9.515386, 0.175208, 0.171846, vec3(-0.000000, -5.000000, 1.510000), vec3(0.005683, 0.488433, -0.106771), vec3(-0.000002, -0.396198, 0.586454), vec3(0.114372, -0.198100, 0.058963), vec3(0.198098, -0.343119, 0.586454));
    TestRayTriangle(1, 9.519246, 0.063619, 0.735248, vec3(-0.000000, -5.000000, 1.510000), vec3(0.037728, 0.497889, -0.026139), vec3(0.228743, -0.396199, 1.307019), vec3(0.343117, -0.198101, 0.586454), vec3(0.396197, -0.228747, 1.307019));
    TestRayTriangle(1, 9.547610, 0.233135, 0.138973, vec3(-0.000000, -5.000000, 1.510000), vec3(0.005676, 0.487804, -0.109611), vec3(-0.000002, -0.396198, 0.586454), vec3(0.114372, -0.198100, 0.058963), vec3(0.198098, -0.343119, 0.586454));
    TestRayTriangle(1, 9.580382, 0.291455, 0.105877, vec3(0.000000, -5.000000, 1.510000), vec3(0.005669, 0.487158, -0.112447), vec3(-0.000002, -0.396198, 0.586454), vec3(0.114372, -0.198100, 0.058963), vec3(0.198098, -0.343119, 0.586454));
    TestRayTriangle(1, 9.584744, 0.065996, 0.914141, vec3(0.000000, -5.000000, 1.510000), vec3(0.040624, 0.497661, -0.026139), vec3(0.228743, -0.396199, 1.307019), vec3(0.343117, -0.198101, 0.586454), vec3(0.396197, -0.228747, 1.307019));
    TestRayTriangle(1, 9.596931, 0.109001, 0.237516, vec3(0.000000, -5.000000, 1.510000), vec3(0.002830, 0.486521, -0.115280), vec3(-0.000002, -0.396198, 0.586454), vec3(-0.000001, -0.228746, 0.058963), vec3(0.114372, -0.198100, 0.058963));
    TestRayTriangle(1, 9.665071, 0.226460, 0.238526, vec3(0.000000, -5.000000, 1.510000), vec3(0.002823, 0.485147, -0.120933), vec3(-0.000002, -0.396198, 0.586454), vec3(-0.000001, -0.228746, 0.058963), vec3(0.114372, -0.198100, 0.058963));
    TestRayTriangle(1, 9.718788, 0.581147, 0.000010, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.483715, -0.126569), vec3(-0.000002, -0.396198, 0.586454), vec3(-0.000001, -0.228746, 0.058963), vec3(0.114372, -0.198100, 0.058963));
    TestRayTriangle(1, 9.754798, 0.641791, 0.000010, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482970, -0.129381), vec3(-0.000002, -0.396198, 0.586454), vec3(-0.000001, -0.228746, 0.058963), vec3(0.114372, -0.198100, 0.058963));
    TestRayTriangle(1, 9.791403, 0.702885, 0.000009, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482209, -0.132189), vec3(-0.000002, -0.396198, 0.586454), vec3(-0.000001, -0.228746, 0.058963), vec3(0.114372, -0.198100, 0.058963));
    TestRayTriangle(1, 9.881036, 0.076744, 0.551589, vec3(0.000000, -5.000000, 1.510000), vec3(0.043518, 0.497416, -0.026139), vec3(0.396197, -0.228747, 1.307019), vec3(0.396198, -0.000002, 0.586454), vec3(0.457490, -0.000002, 1.307019));
    TestRayTriangle(1, 90.192108, 0.475962, 0.107179, vec3(0.000000, -5.000000, 1.510000), vec3(0.078110, 0.493168, -0.026139), vec3(5.454547, 38.181816, -0.815297), vec3(8.181818, 40.909088, -0.822232), vec3(8.181818, 38.181816, -1.085168));
    TestRayTriangle(1, 90.428535, 0.338361, 0.311023, vec3(0.000000, -5.000000, 1.510000), vec3(0.200542, 0.457570, -0.020327), vec3(16.363636, 35.454548, -0.336135), vec3(19.090912, 38.181816, -0.731500), vec3(19.090912, 35.454548, 0.119553));
    TestRayTriangle(1, 90.643234, 0.542164, 0.149211, vec3(0.000000, -5.000000, 1.510000), vec3(0.080978, 0.492706, -0.026139), vec3(5.454547, 38.181816, -0.815297), vec3(8.181818, 40.909088, -0.822232), vec3(8.181818, 38.181816, -1.085168));
    TestRayTriangle(1, 91.101128, 0.608865, 0.191810, vec3(0.000000, -5.000000, 1.510000), vec3(0.083843, 0.492226, -0.026139), vec3(5.454547, 38.181816, -0.815297), vec3(8.181818, 40.909088, -0.822232), vec3(8.181818, 38.181816, -1.085168));
    TestRayTriangle(1, 91.260429, 0.438618, 0.360899, vec3(0.000000, -5.000000, 1.510000), vec3(0.203200, 0.456395, -0.020327), vec3(16.363636, 35.454548, -0.336135), vec3(19.090912, 38.181816, -0.731500), vec3(19.090912, 35.454548, 0.119553));
    TestRayTriangle(1, 91.565178, 0.675966, 0.235071, vec3(0.000000, -5.000000, 1.510000), vec3(0.086705, 0.491730, -0.026139), vec3(5.454547, 38.181816, -0.815297), vec3(8.181818, 40.909088, -0.822232), vec3(8.181818, 38.181816, -1.085168));
    TestRayTriangle(1, 92.024002, 0.719298, 0.022103, vec3(-0.000000, -5.000000, 1.510000), vec3(0.089565, 0.491218, -0.026139), vec3(8.181818, 38.181816, -1.085168), vec3(8.181818, 40.909088, -0.822232), vec3(10.909088, 40.909088, -1.057061));
    TestRayTriangle(1, 92.441368, 0.665993, 0.132622, vec3(0.000000, -5.000000, 1.510000), vec3(0.092421, 0.490688, -0.026139), vec3(8.181818, 38.181816, -1.085168), vec3(8.181818, 40.909088, -0.822232), vec3(10.909088, 40.909088, -1.057061));
    TestRayTriangle(1, 92.543724, 0.497238, 0.074857, vec3(0.000000, -5.000000, 1.510000), vec3(0.208497, 0.454000, -0.020327), vec3(19.090912, 35.454548, 0.119553), vec3(19.090912, 38.181816, -0.731500), vec3(21.818182, 38.181816, -0.782879));
    TestRayTriangle(1, 92.789070, 0.388050, 0.183357, vec3(0.000000, -5.000000, 1.510000), vec3(0.211135, 0.452779, -0.020327), vec3(19.090912, 35.454548, 0.119553), vec3(19.090912, 38.181816, -0.731500), vec3(21.818182, 38.181816, -0.782879));
    TestRayTriangle(1, 93.039276, 0.278328, 0.292467, vec3(-0.000000, -5.000000, 1.510000), vec3(0.213765, 0.451543, -0.020327), vec3(19.090912, 35.454548, 0.119553), vec3(19.090912, 38.181816, -0.731500), vec3(21.818182, 38.181816, -0.782879));
    TestRayTriangle(1, 93.555145, 0.057194, 0.512632, vec3(0.000000, -5.000000, 1.510000), vec3(0.219004, 0.449025, -0.020327), vec3(19.090912, 35.454548, 0.119553), vec3(19.090912, 38.181816, -0.731500), vec3(21.818182, 38.181816, -0.782879));
    TestRayTriangle(1, 94.198341, 0.586670, 0.157551, vec3(-0.000000, -5.000000, 1.510000), vec3(0.224214, 0.446447, -0.020327), vec3(19.090912, 35.454548, 0.119553), vec3(21.818182, 38.181816, -0.782879), vec3(21.818182, 35.454548, 0.151699));
    TestRayTriangle(1, 94.467590, 0.572582, 0.414253, vec3(0.000000, -5.000000, 1.510000), vec3(0.300659, 0.398987, -0.020327), vec3(27.272730, 30.000000, 1.051044), vec3(27.272730, 32.727272, -0.080721), vec3(30.000000, 32.727272, -0.912250));
    TestRayTriangle(1, 94.476532, 0.431853, 0.495471, vec3(-0.000000, -5.000000, 1.510000), vec3(0.302975, 0.397232, -0.020327), vec3(27.272730, 30.000000, 1.051044), vec3(27.272730, 32.727272, -0.080721), vec3(30.000000, 32.727272, -0.912250));
    TestRayTriangle(1, 94.547569, 0.598346, 0.264497, vec3(0.000000, -5.000000, 1.510000), vec3(0.226808, 0.445135, -0.020327), vec3(19.090912, 35.454548, 0.119553), vec3(21.818182, 38.181816, -0.782879), vec3(21.818182, 35.454548, 0.151699));
    TestRayTriangle(1, 94.553162, 0.059760, 0.283274, vec3(0.000000, -5.000000, 1.510000), vec3(0.298333, 0.400730, -0.020327), vec3(27.272730, 32.727272, -0.080721), vec3(30.000000, 35.454548, -1.682980), vec3(30.000000, 32.727272, -0.912250));
    TestRayTriangle(1, 94.622368, 0.282711, 0.024291, vec3(0.000000, -5.000000, 1.510000), vec3(0.288927, 0.407563, -0.020327), vec3(27.272730, 32.727272, -0.080721), vec3(27.272730, 35.454548, -1.119909), vec3(30.000000, 35.454548, -1.682980));
    TestRayTriangle(1, 94.668411, 0.136697, 0.137846, vec3(0.000000, -5.000000, 1.510000), vec3(0.295996, 0.402458, -0.020327), vec3(27.272730, 32.727272, -0.080721), vec3(30.000000, 35.454548, -1.682980), vec3(30.000000, 32.727272, -0.912250));
    TestRayTriangle(1, 94.697449, 0.145239, 0.114404, vec3(-0.000000, -5.000000, 1.510000), vec3(0.291293, 0.405876, -0.020327), vec3(27.272730, 32.727272, -0.080721), vec3(27.272730, 35.454548, -1.119909), vec3(30.000000, 35.454548, -1.682980));
    TestRayTriangle(1, 94.725357, 0.438197, 0.431622, vec3(0.000000, -5.000000, 1.510000), vec3(0.284166, 0.410897, -0.020327), vec3(24.545452, 32.727272, 0.845398), vec3(27.272730, 35.454548, -1.119909), vec3(27.272730, 32.727272, -0.080721));
    TestRayTriangle(1, 94.801140, 0.506845, 0.287606, vec3(0.000000, -5.000000, 1.510000), vec3(0.281770, 0.412543, -0.020327), vec3(24.545452, 32.727272, 0.845398), vec3(27.272730, 35.454548, -1.119909), vec3(27.272730, 32.727272, -0.080721));
    TestRayTriangle(1, 94.880562, 0.575644, 0.143350, vec3(0.000000, -5.000000, 1.510000), vec3(0.279365, 0.414176, -0.020327), vec3(24.545452, 32.727272, 0.845398), vec3(27.272730, 35.454548, -1.119909), vec3(27.272730, 32.727272, -0.080721));
    TestRayTriangle(1, 94.903069, 0.610190, 0.372197, vec3(0.000000, -4.999999, 1.510000), vec3(0.229394, 0.443808, -0.020327), vec3(19.090912, 35.454548, 0.119553), vec3(21.818182, 38.181816, -0.782879), vec3(21.818182, 35.454548, 0.151699));
    TestRayTriangle(1, 94.918648, 0.837364, 0.127967, vec3(0.000000, -5.000000, 1.510000), vec3(0.262271, 0.425206, -0.020327), vec3(24.545452, 32.727272, 0.845398), vec3(24.545452, 35.454548, -0.364777), vec3(27.272730, 35.454548, -1.119909));
    TestRayTriangle(1, 94.919495, 0.558621, 0.299618, vec3(0.000000, -5.000000, 1.510000), vec3(0.267201, 0.422125, -0.020327), vec3(24.545452, 32.727272, 0.845398), vec3(24.545452, 35.454548, -0.364777), vec3(27.272730, 35.454548, -1.119909));
    TestRayTriangle(1, 94.934128, 0.279915, 0.471387, vec3(0.000000, -5.000000, 1.510000), vec3(0.272095, 0.418988, -0.020327), vec3(24.545452, 32.727272, 0.845398), vec3(24.545452, 35.454548, -0.364777), vec3(27.272730, 35.454548, -1.119909));
    TestRayTriangle(1, 94.946739, 0.140533, 0.557345, vec3(0.000000, -5.000000, 1.510000), vec3(0.274527, 0.417398, -0.020327), vec3(24.545452, 32.727272, 0.845398), vec3(24.545452, 35.454548, -0.364777), vec3(27.272730, 35.454548, -1.119909));
    TestRayTriangle(1, 94.996895, 0.498623, 0.080081, vec3(-0.000000, -5.000000, 1.510000), vec3(0.231972, 0.442465, -0.020327), vec3(21.818182, 35.454548, 0.151699), vec3(21.818182, 38.181816, -0.782879), vec3(24.545452, 38.181816, -1.181215));
    TestRayTriangle(1, 95.006935, 0.031985, 0.018143, vec3(0.000000, -5.000000, 1.510000), vec3(0.259793, 0.426724, -0.020327), vec3(24.545452, 35.454548, -0.364777), vec3(27.272730, 38.181816, -1.702072), vec3(27.272730, 35.454548, -1.119909));
    TestRayTriangle(1, 95.064178, 0.145304, 0.736595, vec3(-0.000000, -5.000000, 1.510000), vec3(0.254811, 0.429718, -0.020327), vec3(21.818182, 35.454548, 0.151699), vec3(24.545452, 38.181816, -1.181215), vec3(24.545452, 35.454548, -0.364777));
    TestRayTriangle(1, 95.084045, 0.199856, 0.596589, vec3(-0.000000, -5.000000, 1.510000), vec3(0.252306, 0.431193, -0.020327), vec3(21.818182, 35.454548, 0.151699), vec3(24.545452, 38.181816, -1.181215), vec3(24.545452, 35.454548, -0.364777));
    TestRayTriangle(1, 95.103706, 0.232728, 0.268144, vec3(0.000000, -5.000000, 1.510000), vec3(0.237104, 0.439736, -0.020327), vec3(21.818182, 35.454548, 0.151699), vec3(21.818182, 38.181816, -0.782879), vec3(24.545452, 38.181816, -1.181215));
    TestRayTriangle(1, 95.107430, 0.254503, 0.456478, vec3(0.000000, -5.000000, 1.510000), vec3(0.249794, 0.432654, -0.020327), vec3(21.818182, 35.454548, 0.151699), vec3(24.545452, 38.181816, -1.181215), vec3(24.545452, 35.454548, -0.364777));
    TestRayTriangle(1, 95.134499, 0.309246, 0.316263, vec3(-0.000000, -5.000000, 1.510000), vec3(0.247272, 0.434101, -0.020327), vec3(21.818182, 35.454548, 0.151699), vec3(24.545452, 38.181816, -1.181215), vec3(24.545452, 35.454548, -0.364777));
    TestRayTriangle(1, 95.136848, 0.063631, 0.224893, vec3(-0.000000, -5.000001, 1.510000), vec3(0.092447, 0.490829, -0.023234), vec3(8.181818, 40.909088, -0.822232), vec3(8.181818, 43.636364, -0.281435), vec3(10.909088, 43.636364, -0.433355));
    TestRayTriangle(1, 95.162590, 0.099592, 0.362390, vec3(0.000000, -5.000000, 1.510000), vec3(0.239658, 0.438350, -0.020327), vec3(21.818182, 35.454548, 0.151699), vec3(21.818182, 38.181816, -0.782879), vec3(24.545452, 38.181816, -1.181215));
    TestRayTriangle(1, 95.165321, 0.364086, 0.175948, vec3(0.000000, -5.000000, 1.510000), vec3(0.244742, 0.435532, -0.020327), vec3(21.818182, 35.454548, 0.151699), vec3(24.545452, 38.181816, -1.181215), vec3(24.545452, 35.454548, -0.364777));
    TestRayTriangle(1, 95.199821, 0.419029, 0.035508, vec3(0.000000, -5.000000, 1.510000), vec3(0.242205, 0.436948, -0.020327), vec3(21.818182, 35.454548, 0.151699), vec3(24.545452, 38.181816, -1.181215), vec3(24.545452, 35.454548, -0.364777));
    TestRayTriangle(1, 95.372734, 0.311865, 0.020827, vec3(0.000000, -5.000000, 1.510000), vec3(0.095301, 0.490283, -0.023234), vec3(8.181818, 40.909088, -0.822232), vec3(10.909088, 43.636364, -0.433355), vec3(10.909088, 40.909088, -1.057061));
    TestRayTriangle(1, 95.654106, 0.342730, 0.099778, vec3(-0.000000, -5.000000, 1.510000), vec3(0.098152, 0.489721, -0.023234), vec3(8.181818, 40.909088, -0.822232), vec3(10.909088, 43.636364, -0.433355), vec3(10.909088, 40.909088, -1.057061));
    TestRayTriangle(1, 95.705772, 0.187441, 0.790763, vec3(0.000000, -5.000000, 1.510000), vec3(0.307498, 0.393579, -0.023234), vec3(27.272730, 30.000000, 1.051044), vec3(27.272730, 32.727272, -0.080721), vec3(30.000000, 32.727272, -0.912250));
    TestRayTriangle(1, 95.725380, 0.854504, 0.098473, vec3(0.000000, -5.000000, 1.510000), vec3(0.312057, 0.389975, -0.023234), vec3(27.272730, 30.000000, 1.051044), vec3(30.000000, 32.727272, -0.912250), vec3(30.000000, 30.000000, 0.162953));
    TestRayTriangle(1, 95.725601, 0.044866, 0.873186, vec3(-0.000000, -5.000000, 1.510000), vec3(0.309783, 0.391784, -0.023234), vec3(27.272730, 30.000000, 1.051044), vec3(27.272730, 32.727272, -0.080721), vec3(30.000000, 32.727272, -0.912250));
    TestRayTriangle(1, 95.765739, 0.049448, 0.667493, vec3(-0.000000, -5.000000, 1.510000), vec3(0.305203, 0.395362, -0.023234), vec3(27.272730, 32.727272, -0.080721), vec3(30.000000, 35.454548, -1.682980), vec3(30.000000, 32.727272, -0.912250));
    TestRayTriangle(1, 95.824120, 0.760820, 0.043804, vec3(0.000000, -5.000000, 1.510000), vec3(0.314320, 0.388153, -0.023234), vec3(30.000000, 30.000000, 0.162953), vec3(30.000000, 32.727272, -0.912250), vec3(32.727272, 32.727272, -1.235349));
    TestRayTriangle(1, 95.872910, 0.127154, 0.520741, vec3(0.000000, -5.000000, 1.510000), vec3(0.302898, 0.397130, -0.023234), vec3(27.272730, 32.727272, -0.080721), vec3(30.000000, 35.454548, -1.682980), vec3(30.000000, 32.727272, -0.912250));
    TestRayTriangle(1, 95.939903, 0.373652, 0.179308, vec3(-0.000000, -5.000000, 1.510000), vec3(0.101000, 0.489141, -0.023234), vec3(8.181818, 40.909088, -0.822232), vec3(10.909088, 43.636364, -0.433355), vec3(10.909088, 40.909088, -1.057061));
    TestRayTriangle(1, 96.086830, 0.623896, 0.153454, vec3(-0.000000, -5.000000, 1.510000), vec3(0.316573, 0.386318, -0.023234), vec3(30.000000, 30.000000, 0.162953), vec3(30.000000, 32.727272, -0.912250), vec3(32.727272, 32.727272, -1.235349));
    TestRayTriangle(1, 96.230202, 0.404685, 0.259372, vec3(-0.000000, -5.000000, 1.510000), vec3(0.103844, 0.488545, -0.023234), vec3(8.181818, 40.909088, -0.822232), vec3(10.909088, 43.636364, -0.433355), vec3(10.909088, 40.909088, -1.057061));
    TestRayTriangle(1, 96.823799, 0.466970, 0.421246, vec3(-0.000000, -5.000000, 1.510000), vec3(0.109521, 0.487304, -0.023234), vec3(8.181818, 40.909088, -0.822232), vec3(10.909088, 43.636364, -0.433355), vec3(10.909088, 40.909088, -1.057061));
    TestRayTriangle(1, 97.125778, 0.496694, 0.001240, vec3(-0.000000, -5.000000, 1.510000), vec3(0.112354, 0.486659, -0.023234), vec3(10.909088, 40.909088, -1.057061), vec3(10.909088, 43.636364, -0.433355), vec3(13.636365, 43.636364, -0.506111));
    TestRayTriangle(1, 97.293655, 0.395172, 0.109090, vec3(-0.000000, -5.000000, 1.510000), vec3(0.115183, 0.485996, -0.023234), vec3(10.909088, 40.909088, -1.057061), vec3(10.909088, 43.636364, -0.433355), vec3(13.636365, 43.636364, -0.506111));
    TestRayTriangle(1, 97.464638, 0.293229, 0.217285, vec3(-0.000000, -5.000000, 1.510000), vec3(0.118009, 0.485319, -0.023234), vec3(10.909088, 40.909088, -1.057061), vec3(10.909088, 43.636364, -0.433355), vec3(13.636365, 43.636364, -0.506111));
    TestRayTriangle(1, 97.639229, 0.190843, 0.325851, vec3(-0.000000, -5.000000, 1.510000), vec3(0.120830, 0.484623, -0.023234), vec3(10.909088, 40.909088, -1.057061), vec3(10.909088, 43.636364, -0.433355), vec3(13.636365, 43.636364, -0.506111));
    TestRayTriangle(1, 97.816963, 0.088005, 0.434773, vec3(-0.000000, -5.000000, 1.510000), vec3(0.123648, 0.483912, -0.023234), vec3(10.909088, 40.909088, -1.057061), vec3(10.909088, 43.636364, -0.433355), vec3(13.636365, 43.636364, -0.506111));
    TestRayTriangle(1, 97.992203, 0.527711, 0.016086, vec3(-0.000000, -5.000000, 1.510000), vec3(0.126461, 0.483184, -0.023234), vec3(10.909088, 40.909088, -1.057061), vec3(13.636365, 43.636364, -0.506111), vec3(13.636365, 40.909088, -1.081829));
    TestRayTriangle(1, 98.102783, 0.527633, 0.464814, vec3(0.000000, -5.000000, 1.510000), vec3(0.318724, 0.384359, -0.026139), vec3(30.000000, 30.000000, 0.162953), vec3(30.000000, 32.727272, -0.912250), vec3(32.727272, 32.727272, -1.235349));
    TestRayTriangle(1, 98.136482, 0.526525, 0.125036, vec3(0.000000, -5.000000, 1.510000), vec3(0.129270, 0.482441, -0.023234), vec3(10.909088, 40.909088, -1.057061), vec3(13.636365, 43.636364, -0.506111), vec3(13.636365, 40.909088, -1.081829));
    TestRayTriangle(1, 98.385704, 0.386852, 0.578347, vec3(-0.000000, -5.000000, 1.510000), vec3(0.320954, 0.382498, -0.026139), vec3(30.000000, 30.000000, 0.162953), vec3(30.000000, 32.727272, -0.912250), vec3(32.727272, 32.727272, -1.235349));
    TestRayTriangle(1, 98.434669, 0.523802, 0.344176, vec3(-0.000000, -5.000000, 1.510000), vec3(0.134874, 0.480904, -0.023234), vec3(10.909088, 40.909088, -1.057061), vec3(13.636365, 43.636364, -0.506111), vec3(13.636365, 40.909088, -1.081829));
    TestRayTriangle(1, 98.588631, 0.522284, 0.454362, vec3(0.000000, -5.000000, 1.510000), vec3(0.137670, 0.480111, -0.023234), vec3(10.909088, 40.909088, -1.057061), vec3(13.636365, 43.636364, -0.506111), vec3(13.636365, 40.909088, -1.081829));
    TestRayTriangle(1, 98.674301, 0.245247, 0.692627, vec3(0.000000, -5.000000, 1.510000), vec3(0.323174, 0.380624, -0.026139), vec3(30.000000, 30.000000, 0.162953), vec3(30.000000, 32.727272, -0.912250), vec3(32.727272, 32.727272, -1.235349));
    TestRayTriangle(1, 98.968475, 0.102832, 0.807642, vec3(0.000000, -5.000000, 1.510000), vec3(0.325383, 0.378738, -0.026139), vec3(30.000000, 30.000000, 0.162953), vec3(30.000000, 32.727272, -0.912250), vec3(32.727272, 32.727272, -1.235349));
    TestRayTriangle(1, 99.010971, 0.336877, 0.200430, vec3(0.000000, -5.000000, 1.510000), vec3(0.143247, 0.478477, -0.023234), vec3(13.636365, 40.909088, -1.081829), vec3(13.636365, 43.636364, -0.506111), vec3(16.363636, 43.636364, -0.595334));
    TestRayTriangle(1, 99.207581, 0.874592, 0.041526, vec3(0.000000, -5.000000, 1.510000), vec3(0.327581, 0.376839, -0.026139), vec3(30.000000, 30.000000, 0.162953), vec3(32.727272, 32.727272, -1.235349), vec3(32.727272, 30.000000, -0.395683));
    TestRayTriangle(1, 99.346832, 0.811659, 0.012507, vec3(0.000000, -5.000000, 1.510000), vec3(0.329768, 0.374926, -0.026139), vec3(32.727272, 30.000000, -0.395683), vec3(32.727272, 32.727272, -1.235349), vec3(35.454548, 32.727272, -1.165070));
    TestRayTriangle(1, 99.460609, 0.127509, 0.426724, vec3(0.000000, -5.000000, 1.510000), vec3(0.148804, 0.476778, -0.023234), vec3(13.636365, 40.909088, -1.081829), vec3(13.636365, 43.636364, -0.506111), vec3(16.363636, 43.636364, -0.595334));
    TestRayTriangle(1, 99.691010, 0.021968, 0.540593, vec3(0.000000, -5.000000, 1.510000), vec3(0.151575, 0.475904, -0.023234), vec3(13.636365, 40.909088, -1.081829), vec3(13.636365, 43.636364, -0.506111), vec3(16.363636, 43.636364, -0.595334));
    TestRayTriangle(1, 99.763451, 0.509515, 0.237284, vec3(-0.000000, -5.000000, 1.510000), vec3(0.157102, 0.474108, -0.023234), vec3(13.636365, 40.909088, -1.081829), vec3(16.363636, 43.636364, -0.595334), vec3(16.363636, 40.909088, -0.971854));
    TestRayTriangle(1, 99.780342, 0.478710, 0.369877, vec3(0.000000, -5.000000, 1.510000), vec3(0.159858, 0.473186, -0.023234), vec3(13.636365, 40.909088, -1.081829), vec3(16.363636, 43.636364, -0.595334), vec3(16.363636, 40.909088, -0.971854));

    TestRayTriangle(0, 0.000000, 0.000000, 0.000000, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482209, -0.132189), vec3(-9.762015, 13.875500, 0.800714), vec3(-10.064162, 13.701056, 0.524594), vec3(-9.983202, 13.654313, 0.800714));
    TestRayTriangle(0, 0.000000, 0.000000, 0.000000, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482209, -0.132189), vec3(-9.762015, 13.875500, 0.800714), vec3(-9.808758, 13.956460, 0.524594), vec3(-10.064162, 13.701056, 0.524594));
    TestRayTriangle(0, 0.000000, 0.000000, 0.000000, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482209, -0.132189), vec3(-9.762016, 13.177722, 1.002848), vec3(-9.762018, 12.828833, 0.800714), vec3(-9.634315, 13.050019, 1.002848));
    TestRayTriangle(0, 0.000000, 0.000000, 0.000000, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482209, -0.132189), vec3(-9.762016, 13.177722, 1.002848), vec3(-9.983204, 13.050020, 0.800714), vec3(-9.762018, 12.828833, 0.800714));
    TestRayTriangle(0, 0.000000, 0.000000, 0.000000, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482209, -0.132189), vec3(-9.762016, 13.875500, 0.248473), vec3(-9.634315, 13.654314, 0.046339), vec3(-9.762017, 13.526611, 0.046339));
    TestRayTriangle(0, 0.000000, 0.000000, 0.000000, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482209, -0.132189), vec3(-9.762016, 13.875500, 0.248473), vec3(-9.762017, 13.526611, 0.046339), vec3(-9.983203, 13.654314, 0.248473));
    TestRayTriangle(0, 0.000000, 0.000000, 0.000000, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482209, -0.132189), vec3(-9.762017, 13.526611, 0.046339), vec3(-9.459871, 13.352165, -0.027648), vec3(-9.808760, 13.352166, 0.046339));
    TestRayTriangle(0, 0.000000, 0.000000, 0.000000, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482209, -0.132189), vec3(-9.762017, 13.526611, 0.046339), vec3(-9.459871, 13.352166, -0.027648), vec3(-9.459871, 13.352165, -0.027648));
    TestRayTriangle(0, 0.000000, 0.000000, 0.000000, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482209, -0.132189), vec3(-9.762018, 12.828832, 0.248473), vec3(-9.459870, 13.003275, 0.046339), vec3(-9.459871, 12.747870, 0.248473));
    TestRayTriangle(0, 0.000000, 0.000000, 0.000000, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482209, -0.132189), vec3(-9.762018, 12.828832, 0.248473), vec3(-9.634315, 13.050017, 0.046339), vec3(-9.459870, 13.003275, 0.046339));
    TestRayTriangle(0, 0.000000, 0.000000, 0.000000, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482209, -0.132189), vec3(-9.762018, 12.828833, 0.800714), vec3(-9.459872, 12.654387, 0.524594), vec3(-9.459871, 12.747871, 0.800714));
    TestRayTriangle(0, 0.000000, 0.000000, 0.000000, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482209, -0.132189), vec3(-9.762018, 12.828833, 0.800714), vec3(-9.808761, 12.747872, 0.524594), vec3(-9.459872, 12.654387, 0.524594));
    TestRayTriangle(0, 0.000000, 0.000000, 0.000000, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482209, -0.132189), vec3(-9.762018, 13.177721, 0.046339), vec3(-9.459870, 13.352163, -0.027648), vec3(-9.634315, 13.050017, 0.046339));
    TestRayTriangle(0, 0.000000, 0.000000, 0.000000, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482209, -0.132189), vec3(-9.762018, 13.177721, 0.046339), vec3(-9.459871, 13.352164, -0.027648), vec3(-9.459870, 13.352163, -0.027648));
    TestRayTriangle(0, 0.000000, 0.000000, 0.000000, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482209, -0.132189), vec3(-9.792483, -8.993707, 0.110621), vec3(-9.563810, -9.389782, 0.018547), vec3(-9.792486, -9.618457, 0.018547));
    TestRayTriangle(0, 0.000000, 0.000000, 0.000000, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482209, -0.132189), vec3(-9.792483, -8.993707, 0.110621), vec3(-9.792486, -9.618457, 0.018547), vec3(-10.188560, -9.389782, 0.110621));
    TestRayTriangle(0, 0.000000, 0.000000, 0.000000, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482209, -0.132189), vec3(-9.792483, -8.993709, 0.362173), vec3(-10.333532, -9.306082, 0.236397), vec3(-10.188559, -9.389783, 0.362173));
    TestRayTriangle(0, 0.000000, 0.000000, 0.000000, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482209, -0.132189), vec3(-9.792483, -8.993709, 0.362173), vec3(-9.876184, -8.848734, 0.236397), vec3(-10.333532, -9.306082, 0.236397));
    TestRayTriangle(0, 0.000000, 0.000000, 0.000000, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482209, -0.132189), vec3(-9.792483, -9.618458, 0.454247), vec3(-10.188559, -9.389783, 0.362173), vec3(-10.333533, -9.930831, 0.362173));
    TestRayTriangle(0, 0.000000, 0.000000, 0.000000, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482209, -0.132189), vec3(-9.792483, -9.618458, 0.454247), vec3(-10.333533, -9.930831, 0.362173), vec3(-9.876184, -9.930832, 0.454247));
    TestRayTriangle(0, 0.000000, 0.000000, 0.000000, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482209, -0.132189), vec3(-9.792485, -10.243207, 0.454247), vec3(-10.188561, -10.471880, 0.362173), vec3(-9.792487, -10.867957, 0.362173));
    TestRayTriangle(0, 0.000000, 0.000000, 0.000000, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482209, -0.132189), vec3(-9.792485, -10.243207, 0.454247), vec3(-9.792487, -10.867957, 0.362173), vec3(-9.563812, -10.471882, 0.454247));
    TestRayTriangle(0, 0.000000, 0.000000, 0.000000, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482209, -0.132189), vec3(-9.792486, -9.618457, 0.018547), vec3(-9.251438, -9.930832, -0.015155), vec3(-9.251438, -9.930834, -0.015155));
    TestRayTriangle(0, 0.000000, 0.000000, 0.000000, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482209, -0.132189), vec3(-9.792486, -9.618457, 0.018547), vec3(-9.251438, -9.930834, -0.015155), vec3(-9.876187, -9.930832, 0.018547));
    TestRayTriangle(0, 0.000000, 0.000000, 0.000000, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482209, -0.132189), vec3(-9.792487, -10.867957, 0.362173), vec3(-9.251440, -11.180334, 0.236397), vec3(-9.251439, -11.012932, 0.362173));
    TestRayTriangle(0, 0.000000, 0.000000, 0.000000, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482209, -0.132189), vec3(-9.792487, -10.867957, 0.362173), vec3(-9.876189, -11.012931, 0.236397), vec3(-9.251440, -11.180334, 0.236397));
    TestRayTriangle(0, 0.000000, 0.000000, 0.000000, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482209, -0.132189), vec3(-9.792488, -10.243209, 0.018547), vec3(-9.251437, -9.930837, -0.015155), vec3(-9.563813, -10.471884, 0.018547));
    TestRayTriangle(0, 0.000000, 0.000000, 0.000000, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482209, -0.132189), vec3(-9.792488, -10.243209, 0.018547), vec3(-9.251438, -9.930836, -0.015155), vec3(-9.251437, -9.930837, -0.015155));
    TestRayTriangle(0, 0.000000, 0.000000, 0.000000, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482209, -0.132189), vec3(-9.792488, -10.867958, 0.110621), vec3(-9.251437, -10.555586, 0.018547), vec3(-9.251439, -11.012934, 0.110621));
    TestRayTriangle(0, 0.000000, 0.000000, 0.000000, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482209, -0.132189), vec3(-9.792488, -10.867958, 0.110621), vec3(-9.563813, -10.471884, 0.018547), vec3(-9.251437, -10.555586, 0.018547));
    TestRayTriangle(0, 0.000000, 0.000000, 0.000000, vec3(0.000000, -5.000000, 1.510000), vec3(0.000000, 0.482209, -0.132189), vec3(-9.808758, 13.956460, 0.524594), vec3(-9.762016, 13.875500, 0.248473), vec3(-9.983203, 13.654314, 0.248473));

}

#define TEST(name, fn)                              \
    try { \
        fn(); \
    } catch(const Unimplemented&) { \
        if (!failures++) std::cout << "Failed tests:" <<  std::endl; \
        std::cout << "  " << name << " called unimplemented code." << std::endl; \
    }


void geomlibUnitTests()
{
    std::cout << std::endl;

    TEST("PointLineDistance", PointLineDistance);
    TEST("PointPlaneDistance", PointPlaneDistance);
    TEST("AngleBetweenPlanes", AngleBetweenPlanes);
    TEST("AngleBetweenLines", AngleBetweenLines);
    TEST("AngleBetweenLinePlane", AngleBetweenLinePlane);
    TEST("CoplanarLines", CoplanarLines);
    TEST("ParallelPerpendicular", ParallelPerpendicular);
    TEST("LinePlaneIntersection", LinePlaneIntersection);
    TEST("SegmentPlaneIntersection", SegmentPlaneIntersection);
    TEST("RayPointContainment", RayPointContainment);
    TEST("BoxPointContainment", BoxPointContainment);
    TEST("SegmentPointContainment", SegmentPointContainment);
    TEST("TrianglePointContainment", TrianglePointContainment);
    TEST("SegmentTriangleIntersection", SegmentTriangleIntersection);
    TEST("TriangleIntersection", TriangleIntersection);
    TEST("RaySphereIntersection", RaySphereIntersection);
    TEST("RayBoxIntersection", RayBoxIntersection);
    TEST("RayTriangleIntersection", RayTriangleIntersection);

    std::cout << std::endl;
    std::cout << completed << " tests completed." << std::endl;
    std::cout << failures << " tests failed." << std::endl;
        
    //std::cout << "Press RETURN to quit: ";
    //char x;
    //scanf("%c", &x);
}
