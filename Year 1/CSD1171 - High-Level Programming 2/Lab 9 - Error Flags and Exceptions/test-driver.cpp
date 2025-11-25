#include "test.hpp"

#include <cerrno>		 // errno
#include <exception> // std::exception
#include <iostream>  // std::cin, std::cout
#include <limits>    // std::numeric_limits
#include <sstream>   // std::stringstream
#include <string>    // std::string
#include <utility>   // std::pair

// declarations ...
namespace { // anonymous namespace

// returns an error flag and writes calculated value in output-only reference parameter
bool divide1(int numerator, int denominator, int& value);
// returns an object that aggregates calculated value and error flag
std::pair<bool, int> divide2(int numerator, int denominator);
// returns calculated value and uses global variable errno to indicate whether result is valid
int  divide3(int numerator, int denominator);
// returns calculated value and reserves a specific calculated value to indicate error
int  divide4(int numerator, int denominator);
// returns calculated value and indicates an error by throwing an exception
int  divide5(int numerator, int denominator);

} // end anonymous namespace

int main() {
	try {
	  hlp2::stream_wrapper stream{std::cin};
    int n, d0, d1, d2, d3, d4;
		stream >> n >> d0 >> d1 >> d2 >> d3 >> d4;

		hlp2::test1(n, d0, divide1);
		hlp2::test2(n, d1, divide2);
		hlp2::test3(n, d2, divide3);
		hlp2::test4(n, d3, divide4);
		hlp2::test5(n, d4, divide5);
	} catch (std::exception const& e) {
		std::cout << "\nException was thrown: " << e.what() << std::endl;
	}
}

// definitions ...
namespace {

bool divide1(int numerator, int denominator, int& value) {
	bool success = denominator;
	value = denominator ? (numerator / denominator) : 0;
	return success;
}

std::pair<bool, int> divide2(int numerator, int denominator) {
	bool success = denominator;
	int value = denominator ? (numerator / denominator) : 0;
	return {success, value};
}

int divide3(int numerator, int denominator) {
	errno = denominator ? 1 : 0;
	return denominator ? (numerator / denominator) : 0;
}

int divide4(int numerator, int denominator) {
	if (!denominator) {
		return std::numeric_limits<int>::min();
	}
	return numerator / denominator;
}

int divide5(int numerator, int denominator) {
	if (!denominator) {
		throw hlp2::division_by_zero{numerator};
	}
	return numerator / denominator;
}

} // end anonymous namespace
