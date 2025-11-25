#include <iostream> // std::cin, std::cout, "\n"
#include <array>
#include "point.hpp"  // Point class

// unit tests for operator overloads
namespace {
void DotProductTest(); void ConstructorTest(); void TranslateTest(); void ScaleTest();
void RotateTest();     void DistanceTest();    void MidpointTest();  void InputTest();
void AssignmentTest(); void IncDecTest();      void UnaryTest();     
}

// run test cases
int main() {
  ConstructorTest();
  TranslateTest();
  ScaleTest();
  RotateTest();
  DistanceTest();
  MidpointTest();
  InputTest();
  AssignmentTest();
  IncDecTest();
  UnaryTest();
  DotProductTest();
}

namespace {
void DotProductTest() {
  std::cout << "\n********** Dot Product test ********** " << "\n";
  hlp2::Point const pt1{3., 4.};
  hlp2::Point pt2(1, 2);

  std::cout << "dot product of " << pt1 << " and " << pt2 << " is: " << (pt1*pt2) << "\n";
  std::cout << "dot product of " << (2*pt1) << " and " << pt2 << " is: " << ((2.0*pt1)*pt2) << "\n";
  std::cout << "dot product of " << pt1 << " and " << (pt2*1.1) << " is: " << (pt1*(pt2*1.1)) << "\n";
}

void ConstructorTest() {
  std::cout << "\n********** Constructor test ********** " << "\n";

  hlp2::Point const pt1(3, 4);
  hlp2::Point const pt2;
  hlp2::Point const pt3(pt1);
  hlp2::Point const pt4 = {pt1[0], pt1[1]};
  hlp2::Point pt5;
 
  pt5 = pt4;
  ++pt5[0]; --pt5[1];

  // DO NOT ALLOW THIS SYNTAX:
  //hlp2::Point p6 = 4; // this should not compile
  //hlp2::Point p7(4);  // this should not compile

  std::cout << "Point pt1(3, 4): " << pt1 << "\n";
  std::cout << "Point pt2: " << pt2 << "\n";
  std::cout << "Point pt3(pt1): " << pt3 << "\n";
  std::cout << "Point pt4 = pt1: " << pt3 << "\n";
  std::cout << "pt5 = pt4: " << pt5 << "\n";

  double da[] {1.0, 2.0, 3.0, 4.0};
  hlp2::Point pt6(da);          // pt6 has coordinates (1.0, 2.0)
  hlp2::Point pt7(da+2);        // pt7 has coordinates (3.0, 4.0)
  std::cout << "Point pt6(const double*): " << pt6 << '\n';
  std::cout << "Point pt7(const double*): " << pt7 << '\n';

  std::array<double, 4> sad {1.1, 2.2, 3.3, 4.4};
  hlp2::Point pt8(sad.data());   // pt8 has coordinates (1.1, 2.2)
  hlp2::Point pt9(sad.data()+2); // pt9 has coordinates (3.3, 4.4)
  std::cout << "Point pt8(const double*): " << pt8 << '\n';
  std::cout << "Point pt9(const double*): " << pt9 << '\n';
}

void TranslateTest() {
  std::cout << "\n********** Translate test ********** " << "\n";

  hlp2::Point const pt1(3, 4);
  hlp2::Point const pt2(7, 12);
  hlp2::Point pt3;

  std::cout << pt3 << " += 1 = " << (pt3 += 1) << "\n";
  std::cout << pt3 << " += " << pt1 << " = " << (pt3 += pt1) << "\n";
  std::cout << "++" << pt3 << " += " << pt1 << " = " << ++(pt3 += pt1) << "\n";
  std::cout << pt3 << " += " << pt1 << "++ = " << (pt3 += pt1)++ << "\n";
  std::cout << "pt3: " << pt3 << "\n";

  pt3 = pt1 + pt2;
  std::cout << pt1 << " + " << pt2 << " = " << pt3 << "\n";

  pt3 = pt1 + 5;
  std::cout << pt1 << " + 5 = " << pt3 << "\n";

  pt3 = 7 + pt1;
  std::cout << "7 + " << pt1 << " = " << pt3 << "\n";

  pt3 = pt1 + -pt2;
  std::cout << pt1 << " + -" << pt2 << " = " << pt3 << "\n";

  pt3 = pt1 - pt2;
  std::cout << pt1 << " - " << pt2 << " = " << pt3 << "\n";

  pt3 = pt1 - 2;
  std::cout << pt1 << " - 2 = " << pt3 << "\n";

  pt3 = 2 - pt1;
  std::cout << "2 - " << pt1 << " = " << pt3 << "\n";
}

void ScaleTest() {
  std::cout << "\n********** Scale test ********** " << "\n";
  hlp2::Point const pt1(3, 4), pt2 = pt1 * 3;
  std::cout << pt1 << " * 3 = " << pt2 << "\n";

  hlp2::Point const pt3(5, 8), pt4 = 5 * pt3;
  std::cout << "5 * " << pt3 << " = " << pt4 << "\n";

  std::array<hlp2::Point, 2> pa;
  pa[0][0] = 5; pa[0][1] = 8;
  pa[1] = 5*pa[0];
  for (hlp2::Point const& x : pa) {
    std::cout << x << " * 0.5 = " << (x*0.5) << '\n';
  }
}

void RotateTest() {
  std::cout << "\n********** Rotate test ********** " << "\n";

  hlp2::Point const pt1(-50, -50);
  //double angle = 45.0;
  double angle = 33.51;

  hlp2::Point const pt2 = pt1 % angle;

  std::cout << "Point " << pt1 << " rotated " << angle << " degrees is ";
  std::cout.setf(std::ios_base::fixed, std::ios::floatfield); std::cout.precision(15);            
  std::cout << pt2 << "\n";
  std::cout.unsetf(std::ios_base::fixed); std::cout.precision(6);

  hlp2::Point const pt3 = hlp2::Point(-6, -6);
  //angle = -60.0;
  angle = -66.66;

  hlp2::Point const pt4 = pt3 % angle;
  std::cout << "Point " << pt3 << " rotated " << angle << " degrees is ";
  std::cout.setf(std::ios_base::fixed, std::ios::floatfield); std::cout.precision(15);            
  std::cout << pt4 << "\n";
  std::cout.unsetf(std::ios_base::fixed); std::cout.precision(6);

  //angle = 0.0;
  angle = 1.21;
  std::cout << "Point " << pt1 << " rotated " << angle << " degrees is ";
  std::cout.setf(std::ios_base::fixed, std::ios::floatfield); std::cout.precision(15);            
  std::cout << (pt1%angle) << "\n";
  std::cout.unsetf(std::ios_base::fixed); std::cout.precision(6);

  angle = 90.0;
  std::cout << "Point " << pt1 << " rotated " << angle << " degrees is ";
  std::cout.setf(std::ios_base::fixed, std::ios::floatfield); std::cout.precision(15);            
  std::cout << (pt1%angle) << "\n";
  std::cout.unsetf(std::ios_base::fixed); std::cout.precision(6);

  angle = -90.0;
  std::cout << "Point " << pt1 << " rotated " << angle << " degrees is ";
  std::cout.setf(std::ios_base::fixed, std::ios::floatfield); std::cout.precision(15);            
  std::cout << (pt1%angle) << "\n";
  std::cout.unsetf(std::ios_base::fixed); std::cout.precision(6);

  angle = 360.0+90.0;
  std::cout << "Point " << pt1 << " rotated " << angle << " degrees is ";
  std::cout.setf(std::ios_base::fixed, std::ios::floatfield); std::cout.precision(15);            
  std::cout << (pt1%angle) << "\n";
  std::cout.unsetf(std::ios_base::fixed); std::cout.precision(6);
}

void DistanceTest() {
  std::cout << "\n********** Distance test ********** " << "\n";

  hlp2::Point pt1(3, 4);
  hlp2::Point pt2(7, 12);
  double distance = pt2 / pt1;
  std::cout.setf(std::ios_base::fixed, std::ios::floatfield);
  std::cout.precision(3);
  std::cout << "Distance between " << pt2 << " and " << pt1 << " is " << distance << "\n";
 
  distance = pt1 / pt2;
  std::cout << "Distance between " << pt1 << " and " << pt2 << " is " << distance << "\n";

  pt1 = hlp2::Point(-5, 3);
  pt2 = hlp2::Point(-1, -7);

  distance = pt2 / pt1;
  std::cout << "Distance between " << pt2 << " and " << pt1 << " is " << distance << "\n";

  std::cout.unsetf(std::ios_base::fixed);
  std::cout.precision(6);
}

void MidpointTest() {
  std::cout << "\n********** Midpoint test ********** " << "\n";

  hlp2::Point pt1(3, 4);
  hlp2::Point pt2(7, 12);
  hlp2::Point pt3 = pt1 ^ pt2;
  std::cout.setf(std::ios_base::fixed, std::ios::floatfield);
  std::cout.precision(3);
  std::cout << "Midpoint of " << pt1 << " and " << pt2 << " is " << pt3 << "\n";

  pt1 = hlp2::Point(-5, 3);
  pt2 = hlp2::Point(-1, -7);
  pt3 = pt1 ^ pt2;
  std::cout << "Midpoint of " << pt1 << " and " << pt2 << " is " << pt3 << "\n";

  std::cout.unsetf(std::ios_base::fixed);
  std::cout.precision(6);
}

void InputTest() {
  std::cout << "\n********** Input test ********** " << "\n";

  hlp2::Point pt1, pt2;
  std::cout << "Enter an X Y pair (e.g. 3 7): ";
  std::cin >> pt1;
  std::cout << "\npt1: " << pt1 << "\n";
  std::cout << "Enter an X Y pair (e.g. 3 7): ";
  std::cin >> pt2;
  std::cout << "\npt2: " << pt2 << "\n";
  
  hlp2::Point pt3, pt4, pt5;
  std::cout << "Enter an X Y pair (e.g. 3 7): ";
  std::cin >> pt3 >> pt4 >> pt5;
  std::cout << "\npt3: " << pt3 << "\n";
  std::cout << "pt4: " << pt4 << "\n";
  std::cout << "pt5: " << pt5 << "\n";

  std::cout << "Midpoint of " << pt1 << " and " << pt2 << " is " << (pt1 ^ pt2) << "\n";
  std::cout << "Distance between " << pt3 << " and " << pt5 << " is " << (pt3 / pt5) << "\n";
}

void AssignmentTest() {
  std::cout << "\n********** Assignment test ********** " << "\n";

  hlp2::Point pt1(3, 4);
  hlp2::Point pt2(-5, 1);

  std::cout << "pt1 = " << pt1 << "\n";
  std::cout << "pt2 = " << pt2 << "\n";
  pt1 += pt2;
  std::cout << "pt1 += pt2" << "\n";
  std::cout << "pt1 = " << pt1 << "\n";
  std::cout << "pt2 = " << pt2 << "\n";
  std::cout << "\n";

  hlp2::Point pt3(3, 4);
  std::cout << "pt3 = " << pt3 << "\n";
  std::cout << "pt3 += 5" << "\n";
  pt3 += 5;
  std::cout << "pt3 = " << pt3 << "\n";
  std::cout << "\n";

  pt1 += pt2 += pt3;
  std::cout << "pt1 += pt2 += pt3" << "\n";
  std::cout << "pt1 = " << pt1 << "\n";
  std::cout << "pt2 = " << pt2 << "\n";
  std::cout << "pt3 = " << pt3 << "\n";
}

void IncDecTest() {
  std::cout << "\n********** Inc/Dec test ********** " << "\n";

  hlp2::Point pt1(3, 4);
  std::cout << "pt1 = " << pt1 << "\n";

  hlp2::Point pt2 = pt1++;
  std::cout << "pt2 = pt1++" << "\n";
  std::cout << "pt1 = " << pt1 << "\n";
  std::cout << "pt2 = " << pt2 << "\n";

  hlp2::Point pt3(6, 8);
  std::cout << "pt3 = " << pt3 << "\n";

  hlp2::Point pt4 = ++pt3;
  std::cout << "pt4 = ++pt3" << "\n";
  std::cout << "pt3 = " << pt3 << "\n";
  std::cout << "pt4 = " << pt4 << "\n";
  std::cout << "\n";

  hlp2::Point pt5(3, 4);
  std::cout << "pt5 = " << pt5 << "\n";

  hlp2::Point pt6 = pt5--;
  std::cout << "pt6 = pt5--" << "\n";
  std::cout << "pt5 = " << pt5 << "\n";
  std::cout << "pt6 = " << pt6 << "\n";

  hlp2::Point pt7(6, 8);
  std::cout << "pt7 = " << pt7 << "\n";

  hlp2::Point pt8 = --pt7;
  std::cout << "pt8 = --pt7" << "\n";
  std::cout << "pt7 = " << pt7 << "\n";
  std::cout << "pt8 = " << pt8 << "\n";
  std::cout << "\n";

  hlp2::Point pt9 = (--pt7)++; 
  std::cout << "pt9 = (--pt7)++" << "\n";
  std::cout << "pt7 = " << pt7 << "\n";
  std::cout << "pt9 = " << pt9 << "\n";

  hlp2::Point pt10 = (++pt7)--; 
  std::cout << "pt10 = (++pt7)--" << "\n";
  std::cout << "pt7 = " << pt7 << "\n";
  std::cout << "pt9 = " << pt9 << "\n";
  std::cout << "pt10 = " << pt10 << "\n";

  // These should fail to compile - if they compile, your code is wrong.
  //hlp2::Point pt21 = ++(pt7--); 
  //hlp2::Point pt22 = --(pt7++); 
  //hlp2::Point pt11 = ++( pt7 % 45.0 );
  //hlp2::Point pt12 = ++( pt7 ^ pt8 );
  //hlp2::Point pt13 = ++( pt7 - 5.0 );
  //hlp2::Point pt14 = ++( -pt7 );
  //hlp2::Point pt15 = ++( pt7 + pt8 );
  //hlp2::Point pt16 = ++( pt7 + 5.0 );
  //hlp2::Point pt17 = ++( pt7 * 5.0 );
  //hlp2::Point pt18 = ++( 5.0 + pt7 );
  //hlp2::Point pt19 = ( 5.0 * pt7 )++;
}

void UnaryTest() {
  std::cout << "\n********** Unary test ********** " << "\n";

  hlp2::Point pt1(3, 4);
  std::cout << "pt1 = " << pt1 << "\n";
  hlp2::Point pt2 = -pt1;
  std::cout << "pt2 = -pt1" << "\n";

  std::cout << "pt1 = " << pt1 << "\n";
  std::cout << "pt2 = " << pt2 << "\n";
  std::cout << "\n";

  hlp2::Point pt3(-3, 4);
  std::cout << "pt3 = " << pt3 << "\n";
  hlp2::Point pt4 = - - -pt3;
  std::cout << "pt4 = - - -pt3" << "\n";

  std::cout << "pt3 = " << pt3 << "\n";
  std::cout << "pt4 = " << pt4 << "\n";
}
} // end anonymous namespace
