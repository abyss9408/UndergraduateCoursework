#include <iostream> // cout, endl
#include <iomanip>  // setw
#include <string>   // string, stoi
#include <array>    // array
#include "ocean.h"
#include "PRNG.h"

// declarations ...
namespace {
void Dump(HLP2::WarBoats::Ocean const& ocean,
        int field_width = 4,  bool extraline = false, bool showboats = true);
void PrintShotResult(HLP2::WarBoats::ShotResult const& sr);
void PrintStats(HLP2::WarBoats::Ocean const& ocean);
void TakeAShot(HLP2::WarBoats::Ocean& ocean, HLP2::WarBoats::Point const& pt);
void test1();
void test2();
void test3();
void test4();
void test5();
} // end anonymous namespace

// driver ...
int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: ./ocean2.out integer-value-between-1-and-5\n";
    return 0;
  }

  using PtrFunc = void (*)();
  int constexpr MAX_TESTS {5};
  std::array<PtrFunc, MAX_TESTS> tests {test1, test2, test3, test4, test5};

  int test_num { std::stoi(std::string{argv[1]})};
  test_num = (test_num < 0) ? 0 : (test_num > MAX_TESTS) ? MAX_TESTS : test_num;
  tests[test_num-1]();
}

// definitions of private stuff ...
namespace {
void Dump(const HLP2::WarBoats::Ocean &ocean, int field_width, 
          bool extraline, bool showboats) {
  int width {ocean.GetDimensions().x};
  int height {const_cast<HLP2::WarBoats::Ocean const&>(ocean).GetDimensions().y}; // get ocean size 

  for (int y {0}; y < height; y++) {
    for (int x {0}; x < width; x++) {
      int value {const_cast<HLP2::WarBoats::Ocean const&>(ocean).GetGrid()[y * width + x]};
      if ( (value > 0) && (value < HLP2::WarBoats::HIT_OFFSET) 
           && (showboats == false) ) { value = 0; }
      std::cout << std::setw(field_width) << value;
    }
    std::cout << "\n";
    if (extraline) { std::cout << "\n"; }
  }
}

void PrintShotResult(HLP2::WarBoats::ShotResult const& sr) {
  std::array<std::string, 5> msgs { "HIT", "MISS", "DUPLICATE", "SUNK", "ILLEGAL"};
  std::cout << msgs[static_cast<int>(sr)] << "\n";
}

void PrintStats(HLP2::WarBoats::Ocean const& ocean) {
  HLP2::WarBoats::ShotStats stats = const_cast<HLP2::WarBoats::Ocean const&>(ocean).GetShotStats();
  std::cout << "      Hits: " << std::setw(3) << stats.hits << "\n";
  std::cout << "    Misses: " << std::setw(3) << stats.misses << "\n";
  std::cout << "Duplicates: " << std::setw(3) << stats.duplicates << "\n";
  std::cout << "Boats Sunk: " << std::setw(3) << stats.sunk << "\n";
}

void TakeAShot(HLP2::WarBoats::Ocean& ocean, HLP2::WarBoats::Point const& pt) {
  HLP2::WarBoats::ShotResult sr;
  std::cout << "Shot: " << pt.x << ", " << pt.y << "  ";
  sr = ocean.TakeShot(pt);
  PrintShotResult(sr);
  PrintStats(ocean);
  Dump(ocean);
  std::cout << "\n";
}

void test1() {
  HLP2::Utils::srand(0, 0);  // Initialize the pseudo-random number generator
  
  // setup the ocean
  int num_boats {3}, xsize {8}, ysize {8};
  HLP2::WarBoats::Ocean ocean(num_boats, xsize, ysize);

  std::cout << "The empty board" << "\n"; Dump(ocean); std::cout << "\n";

  HLP2::WarBoats::Boat boat;
  HLP2::WarBoats::Point location;
  HLP2::WarBoats::BoatPlacement bp;

    // Boat #1
  boat.ID = 1;
  location = {1, 3};
  boat.position = location;
  boat.orientation = HLP2::WarBoats::oHORIZONTAL;
  bp = const_cast<HLP2::WarBoats::Ocean const&>(ocean).PlaceBoat(boat);
  std::cout << "Boat placement is " << ((HLP2::WarBoats::bpREJECTED == bp)? "rejected.\n" : "accepted.\n");

    // Boat #2
  boat.ID = 2;
  location = {5, 1};
  boat.position = location;
  boat.orientation = HLP2::WarBoats::oVERTICAL;
  bp = ocean.PlaceBoat(boat);
  std::cout << "Boat placement is " << ((HLP2::WarBoats::bpREJECTED == bp)? "rejected.\n" : "accepted.\n");

    // Boat #3
  boat.ID = 3;
  location = {0, 5};
  boat.position = location;
  boat.orientation = HLP2::WarBoats::oHORIZONTAL;
  bp = ocean.PlaceBoat(boat);
  std::cout << "Boat placement is " << ((HLP2::WarBoats::bpREJECTED == bp)? "rejected.\n" : "accepted.\n");

    // Illegal placement
  boat.ID = 4;
  location = {10, 5};
  boat.position = location;
  boat.orientation = HLP2::WarBoats::oHORIZONTAL;
  bp = ocean.PlaceBoat(boat);
  std::cout << "Boat placement is " << ((HLP2::WarBoats::bpREJECTED == bp)? "rejected.\n" : "accepted.\n");

    // Illegal placement
  boat.ID = 4;
  location = {2, 0};
  boat.position = location;
  boat.orientation = HLP2::WarBoats::oVERTICAL;
  bp = ocean.PlaceBoat(boat);
  std::cout << "Boat placement is " << ((HLP2::WarBoats::bpREJECTED == bp)? "rejected.\n" : "accepted.\n");

      // Illegal placement
  boat.ID = 4;
  location = {5, 6};
  boat.position = location;
  boat.orientation = HLP2::WarBoats::oHORIZONTAL;
  bp = ocean.PlaceBoat(boat);
  std::cout << "Boat placement is " << ((HLP2::WarBoats::bpREJECTED == bp)? "rejected.\n" : "accepted.\n");

  std::cout << "The board with " << num_boats << " boats" << "\n"; Dump(ocean, 4, true); std::cout << "\n";
}

void test2() {
  HLP2::Utils::srand(0, 0); // Initialize the pseudo-random number generator

    // Setup the ocean
  int num_boats {3}, xsize {8}, ysize {8};
  HLP2::WarBoats::Ocean ocean(num_boats, xsize, ysize);

  std::cout << "The empty board" << "\n"; Dump(ocean); std::cout << "\n";

  HLP2::WarBoats::Boat boat;

    // Boat #1
  boat.ID = 1;
  boat.position = {1, 3};
  boat.orientation = HLP2::WarBoats::oHORIZONTAL;
  ocean.PlaceBoat(boat);

    // Boat #2
  boat.ID = 2;
  boat.position = {5, 1};
  boat.orientation = HLP2::WarBoats::oVERTICAL;
  ocean.PlaceBoat(boat);

    // Boat #3
  boat.ID = 3;
  boat.position = {0, 5};
  boat.orientation = HLP2::WarBoats::oHORIZONTAL;
  ocean.PlaceBoat(boat);

  std::cout << "The board with " << num_boats << " boats" << "\n"; Dump(ocean); std::cout << "\n";

  TakeAShot(ocean, HLP2::WarBoats::Point{10, 5}); // Illegal coordinate
  TakeAShot(ocean, HLP2::WarBoats::Point{1, 3}); // HIT
  TakeAShot(ocean, HLP2::WarBoats::Point{5, 7}); // MISS
  TakeAShot(ocean, HLP2::WarBoats::Point{1, 3}); // DUPLICATE
  TakeAShot(ocean, HLP2::WarBoats::Point{2, 3}); // HIT
  TakeAShot(ocean, HLP2::WarBoats::Point{3, 3}); // HIT
  TakeAShot(ocean, HLP2::WarBoats::Point{4, 3}); // SUNK
}

void test3() {
  HLP2::Utils::srand(0, 0); // Initialize the pseudo-random number generator

  int num_boats {5}, xsize {10}, ysize {10}; // Setup the ocean (check 100, 10x10)
    // Use a pointer to an ocean and dynamically create it
  HLP2::WarBoats::Ocean *ocean = new HLP2::WarBoats::Ocean(num_boats, xsize, ysize);

  std::cout << "The empty board" << "\n"; Dump(*ocean); std::cout << "\n";

  HLP2::WarBoats::Boat boat;
    // Boat #1
  boat.ID = 1;
  boat.position = {1, 3};
  boat.orientation = HLP2::WarBoats::oHORIZONTAL;
  ocean->PlaceBoat(boat);

    // Boat #2
  boat.ID = 2;
  boat.position = {5, 1};
  boat.orientation = HLP2::WarBoats::oVERTICAL;
  ocean->PlaceBoat(boat);

    // Boat #3
  boat.ID = 3;
  boat.position = {0, 5};
  boat.orientation = HLP2::WarBoats::oHORIZONTAL;
  ocean->PlaceBoat(boat);

    // Boat #4
  boat.ID = 4;
  boat.position = {3, 8};
  boat.orientation = HLP2::WarBoats::oHORIZONTAL;
  ocean->PlaceBoat(boat);

    // Boat #5
  boat.ID = 5;
  boat.position = {9, 6};
  boat.orientation = HLP2::WarBoats::oVERTICAL;
  ocean->PlaceBoat(boat);

  std::cout << "The board with " << num_boats << " boats" << "\n"; Dump(*ocean); std::cout << "\n";

    // Try to sink the boats with random shots
  int shots {0};
  while (num_boats > 0) {
    HLP2::WarBoats::ShotResult sr;
    do {
      HLP2::WarBoats::Point coordinate;
      coordinate.x = HLP2::Utils::Random(0, xsize - 1);
      coordinate.y = HLP2::Utils::Random(0, ysize - 1);
      std::cout << "Shot #" << ++shots << " at coordinate "
                << coordinate.x << ", " << coordinate.y << "  ";
      sr = ocean->TakeShot(coordinate);
      PrintShotResult(sr);
    } while (HLP2::WarBoats::srDUPLICATE == sr);

      // Sunk a boat
    num_boats = (HLP2::WarBoats::srSUNK == sr) ? num_boats-1 : num_boats;

    PrintStats(*ocean);
    Dump(*ocean); std::cout << "\n";
  }

  delete ocean; // Must delete the ocean "manually"
}

void test4() {
  HLP2::Utils::srand(0, 0); // Initialize the pseudo-random number generator
  int num_boats {1}, xsize {5}, ysize {5}; // Setup the ocean

  HLP2::WarBoats::Ocean *ocean = new HLP2::WarBoats::Ocean(num_boats, xsize, ysize);

  std::cout << "The empty board" << "\n"; Dump(*ocean); std::cout << "\n";

  HLP2::WarBoats::Boat boat;
  HLP2::WarBoats::BoatPlacement bp;

    // Illegal placement
  boat.ID = 1;
  boat.position = {-1, 0};
  boat.orientation = HLP2::WarBoats::oVERTICAL;
  bp = ocean->PlaceBoat(boat);
  std::cout << "Boat placement is " << ((HLP2::WarBoats::bpREJECTED == bp)? "rejected.\n" : "accepted.\n");

    // Illegal placement
  boat.ID = 1;
  boat.position = {-10, 0};
  boat.orientation = HLP2::WarBoats::oVERTICAL;
  bp = ocean->PlaceBoat(boat);
  std::cout << "Boat placement is " << ((HLP2::WarBoats::bpREJECTED == bp)? "rejected.\n" : "accepted.\n");

    // Illegal placement
  boat.ID = 1;
  boat.position = {0, -1};
  boat.orientation = HLP2::WarBoats::oVERTICAL;
  bp = ocean->PlaceBoat(boat);
  std::cout << "Boat placement is " << ((HLP2::WarBoats::bpREJECTED == bp)? "rejected.\n" : "accepted.\n");

    // Illegal placement
  boat.ID = 1;
  boat.position = {0, -10};
  boat.orientation = HLP2::WarBoats::oVERTICAL;
  std::cout << "Boat placement is " << ((HLP2::WarBoats::bpREJECTED == bp)? "rejected.\n" : "accepted.\n");

    // Illegal placement
  boat.ID = 1;
  boat.position = {6, 0};
  boat.orientation = HLP2::WarBoats::oVERTICAL;
  bp = ocean->PlaceBoat(boat);
  std::cout << "Boat placement is " << ((HLP2::WarBoats::bpREJECTED == bp)? "rejected.\n" : "accepted.\n");

    // Illegal placement
  boat.ID = 1;
  boat.position = {100, 0};
  boat.orientation = HLP2::WarBoats::oVERTICAL;
  bp = ocean->PlaceBoat(boat);
  std::cout << "Boat placement is " << ((HLP2::WarBoats::bpREJECTED == bp)? "rejected.\n" : "accepted.\n");

      // Illegal placement
  boat.ID = 1;
  boat.position = {0, 6};
  boat.orientation = HLP2::WarBoats::oVERTICAL;
  bp = ocean->PlaceBoat(boat);
  std::cout << "Boat placement is " << ((HLP2::WarBoats::bpREJECTED == bp)? "rejected.\n" : "accepted.\n");

    // Illegal placement
  boat.ID = 1;
  boat.position = {0, 100};
  boat.orientation = HLP2::WarBoats::oVERTICAL;
  bp = ocean->PlaceBoat(boat);
  std::cout << "Boat placement is " << ((HLP2::WarBoats::bpREJECTED == bp)? "rejected.\n" : "accepted.\n");

    // Legal placement
  boat.ID = 1;
  boat.position = {0, 0};
  boat.orientation = HLP2::WarBoats::oVERTICAL;
  bp = ocean->PlaceBoat(boat);
  std::cout << "Boat placement is " << ((HLP2::WarBoats::bpREJECTED == bp)? "rejected.\n" : "accepted.\n");

  std::cout << "The board with " << num_boats << " boats" << "\n"; Dump(*ocean, 4, true); std::cout << "\n";

  TakeAShot(*ocean, HLP2::WarBoats::Point{-1, 0}); // Illegal coordinate
  TakeAShot(*ocean, HLP2::WarBoats::Point{-100, 0});
  TakeAShot(*ocean, HLP2::WarBoats::Point{0, -1});
  TakeAShot(*ocean, HLP2::WarBoats::Point{0, -100});
  TakeAShot(*ocean, HLP2::WarBoats::Point{6, 0});
  TakeAShot(*ocean, HLP2::WarBoats::Point{60, 0});
  TakeAShot(*ocean, HLP2::WarBoats::Point{0, 6});
  TakeAShot(*ocean, HLP2::WarBoats::Point{0, 60});

  delete ocean;
}

void test5() {
  HLP2::Utils::srand(0, 0); // Initialize the pseudo-random number generator

  int num_boats {5}, xsize {8}, ysize {12};
  bool show_sunk {false}, show_all {false};
  HLP2::WarBoats::Ocean ocean(num_boats, xsize, ysize);

    // Place the boats randomly in the ocean
  int boats_placed {0};
  while (boats_placed < num_boats) {
    HLP2::WarBoats::Boat boat;

    //boat.hits = 0;
    boat.ID = boats_placed + 1;
    HLP2::WarBoats::BoatPlacement bp;
    do {
        // Pick a random orientation
      boat.orientation = HLP2::Utils::Random(0, 1)
                                 ? HLP2::WarBoats::oHORIZONTAL
                                 : HLP2::WarBoats::oVERTICAL;

        // Pick a random location
      HLP2::WarBoats::Point location;
      location.x = HLP2::Utils::Random(0, xsize - 1);
      location.y = HLP2::Utils::Random(0, ysize - 1);
      boat.position = location;

        // Place the boat
      bp = ocean.PlaceBoat(boat);

    } while (bp == HLP2::WarBoats::bpREJECTED);

    //ocean.Dump();
    //std::cout << std::endl;
    boats_placed++;
  }
  std::cout << "Boats placed:\n";
  Dump(ocean);
  std::cout << std::endl;

    // Try to sink the boats with random shots
  while (num_boats > 0) {
    HLP2::WarBoats::ShotResult sr;
    do {
      HLP2::WarBoats::Point coordinate;
      coordinate.x = HLP2::Utils::Random(0, xsize - 1);
      coordinate.y = HLP2::Utils::Random(0, ysize - 1);
      //std::cout << "target: " << coordinate.x << ", "
      //          << coordinate.y << std::endl;

      sr = ocean.TakeShot(coordinate);
    } while (sr == HLP2::WarBoats::srDUPLICATE);

      // Sunk a boat
    if (sr == HLP2::WarBoats::srSUNK)
      num_boats--;

    if (show_all) {
      PrintShotResult(sr);
      PrintStats(ocean);
      Dump(ocean, 4, false);
      std::cout << std::endl;
    } else if ( (sr == HLP2::WarBoats::srSUNK) && show_sunk) {
      PrintShotResult(sr);
      PrintStats(ocean);
      Dump(ocean, 4, false);
      std::cout << std::endl;
    }
  }
  std::cout << "Final board:\n";
  PrintStats(ocean);
  Dump(ocean, 4, false);
  std::cout << std::endl;
}
} // end anonymous namespace