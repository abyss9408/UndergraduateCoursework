#include <iostream> // std::cout
#include <iomanip>  // std::setw
#include <string>
#include <array>
#include "ocean.h"
#include "PRNG.h"

// declarations in anonymous namespace
namespace {

void test1();
void test2();
void test3();
void test_rect(int num_boats, int xsize, int ysize, 
              bool show_sunk = true, bool show_all = false);
              
} // end anonymous namespace

// definitions
int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: ./ocean.out test-number-between-1-and-6\n";
    return 0;
  }

  int choice = std::stoi(argv[1]);
  choice = (choice < 1) ? 1 : (choice > 6) ? 6 : choice;
  switch (choice) {
    case 1: test1(); break;
    case 2: test2(); break;
    case 3: test3(); break;
    // usage: test_rect(boats, xsize, ysize, show_sunk, show_all)
    case 4: test_rect(5,  10, 10, true, true); break;
    case 5: test_rect(5,  8,  12, true, true); break;
    case 6: test_rect(99, 30, 30, true, true); break;
  }
}

// definitions in anonymous namespace
namespace {

void PrintShotResult(HLP2::WarBoats::ShotResult const& sr) {
  static std::array<std::string, 5> shot_results {"HIT", "MISS", "DUPLICATE", "SUNK", "ILLEGAL"};
  std::cout << shot_results[static_cast<int>(sr)] << "\n";
}

void PrintStats(HLP2::WarBoats::ShotStats const& stats) {
  std::cout << "      Hits: " << std::setw(3) << stats.hits << "\n";
  std::cout << "    Misses: " << std::setw(3) << stats.misses << "\n";
  std::cout << "Duplicates: " << std::setw(3) << stats.duplicates << "\n";
  std::cout << "Boats Sunk: " << std::setw(3) << stats.sunk << "\n";
}

void TakeAShot(HLP2::WarBoats::Ocean& ocean, HLP2::WarBoats::Point const& pt) {
  std::cout << "Shot: " << pt.x << ", " << pt.y << "  ";
  HLP2::WarBoats::ShotResult sr = HLP2::WarBoats::TakeShot(ocean, pt);
  PrintShotResult(sr);
  PrintStats(GetShotStats(ocean));
  HLP2::WarBoats::DumpOcean(ocean);
  std::cout << "\n";
}

void test1() {
    // Initialize the pseudo-random number generator
  HLP2::Utils::srand(0, 0);

    // Setup the ocean
  int num_boats {3}, xsize {8}, ysize {8};
  HLP2::WarBoats::Ocean *theOcean = HLP2::WarBoats::CreateOcean(num_boats, xsize, ysize);
  HLP2::WarBoats::Ocean &ocean = *theOcean;

  std::cout << "The empty board" << "\n";
  HLP2::WarBoats::DumpOcean(ocean);
  std::cout << "\n";

  HLP2::WarBoats::Boat boat;
  HLP2::WarBoats::BoatPlacement bp;

    // Boat #1
  boat.ID = 1;
  boat.position = {1, 3};
  boat.orientation = HLP2::WarBoats::oHORIZONTAL;
  bp = HLP2::WarBoats::PlaceBoat(ocean, boat);
  std::cout << "Boat placement is " 
            << ((bp == HLP2::WarBoats::bpREJECTED) ? "rejected.\n" : "accepted.\n");

    // Boat #2
  boat.ID = 2;
  boat.position = {5, 1};
  boat.orientation = HLP2::WarBoats::oVERTICAL;
  bp = HLP2::WarBoats::PlaceBoat(ocean, boat);
  std::cout << "Boat placement is " 
            << ((bp == HLP2::WarBoats::bpREJECTED) ? "rejected.\n" : "accepted.\n");

    // Boat #3
  boat.ID = 3;
  boat.position = {0, 5};
  boat.orientation = HLP2::WarBoats::oHORIZONTAL;
  bp = HLP2::WarBoats::PlaceBoat(ocean, boat);
  std::cout << "Boat placement is " 
            << ((bp == HLP2::WarBoats::bpREJECTED) ? "rejected.\n" : "accepted.\n");

    // Illegal placement
  boat.ID = 4;
  boat.position = {10, 5};
  boat.orientation = HLP2::WarBoats::oHORIZONTAL;
  bp = HLP2::WarBoats::PlaceBoat(ocean, boat);
  std::cout << "Boat placement is " 
            << ((bp == HLP2::WarBoats::bpREJECTED) ? "rejected.\n" : "accepted.\n");

    // Illegal placement
  boat.ID = 4;
  boat.position = {2, 0};
  boat.orientation = HLP2::WarBoats::oVERTICAL;
  bp = HLP2::WarBoats::PlaceBoat(ocean, boat);
  std::cout << "Boat placement is " 
            << ((bp == HLP2::WarBoats::bpREJECTED) ? "rejected.\n" : "accepted.\n");

      // Illegal placement
  boat.ID = 4;
  boat.position = {5, 6};
  boat.orientation = HLP2::WarBoats::oHORIZONTAL;
  bp = HLP2::WarBoats::PlaceBoat(ocean, boat);
  std::cout << "Boat placement is " 
            << ((bp == HLP2::WarBoats::bpREJECTED) ? "rejected.\n" : "accepted.\n");

  std::cout << "The board with " << num_boats << " boats" << "\n";
  HLP2::WarBoats::DumpOcean(ocean, 4, true);
  std::cout << "\n";
  HLP2::WarBoats::DestroyOcean(theOcean);
}

void test2() {
    // Initialize the pseudo-random number generator
  HLP2::Utils::srand(0, 0);

    // Setup the ocean
  int num_boats{3}, xsize{8}, ysize{8};
  HLP2::WarBoats::Ocean *theOcean = HLP2::WarBoats::CreateOcean(num_boats, xsize, ysize);
  HLP2::WarBoats::Ocean &ocean = *theOcean;

  std::cout << "The empty board" << "\n";
  HLP2::WarBoats::DumpOcean(ocean);
  std::cout << "\n";

  HLP2::WarBoats::Boat boat;

    // Boat #1
  boat.ID = 1;
  boat.position = {1, 3};
  boat.orientation = HLP2::WarBoats::oHORIZONTAL;
  HLP2::WarBoats::PlaceBoat(ocean, boat);

    // Boat #2
  boat.ID = 2;
  boat.position = {5, 1};
  boat.orientation = HLP2::WarBoats::oVERTICAL;
  HLP2::WarBoats::PlaceBoat(ocean, boat);

    // Boat #3
  boat.ID = 3;
  boat.position = {0, 5};
  boat.orientation = HLP2::WarBoats::oHORIZONTAL;
  HLP2::WarBoats::PlaceBoat(ocean, boat);

  std::cout << "The board with " << num_boats << " boats" << "\n";
  HLP2::WarBoats::DumpOcean(ocean);
  std::cout << "\n";
    
  TakeAShot(ocean, {10, 5}); // Illegal coordinate
  TakeAShot(ocean, {1, 3}); // HIT
  TakeAShot(ocean, {5, 7}); // MISS
  TakeAShot(ocean, {1, 3}); // DUPLICATE
  TakeAShot(ocean, {2, 3}); // HIT
  TakeAShot(ocean, {3, 3}); // HIT
  TakeAShot(ocean, {4, 3}); // SUNK
  HLP2::WarBoats::DestroyOcean(theOcean);
}

void test3() {
    // Initialize the pseudo-random number generator
  HLP2::Utils::srand(0, 0);

    // Setup the ocean (check 100, 30x30)
  int num_boats{5}, xsize{10}, ysize{10};
  HLP2::WarBoats::Ocean *theOcean = HLP2::WarBoats::CreateOcean(num_boats, xsize, ysize);
  HLP2::WarBoats::Ocean &ocean = *theOcean;

  std::cout << "The empty board" << "\n";
  HLP2::WarBoats::DumpOcean(ocean);
  std::cout << "\n";

  HLP2::WarBoats::Boat boat;

    // Boat #1
  boat.ID = 1;
  boat.position = {1, 3};
  boat.orientation = HLP2::WarBoats::oHORIZONTAL;
  HLP2::WarBoats::PlaceBoat(ocean, boat);

    // Boat #2
  boat.ID = 2;
  boat.position = {5, 1};
  boat.orientation = HLP2::WarBoats::oVERTICAL;
  HLP2::WarBoats::PlaceBoat(ocean, boat);

    // Boat #3
  boat.ID = 3;
  boat.position = {0, 5};
  boat.orientation = HLP2::WarBoats::oHORIZONTAL;
  HLP2::WarBoats::PlaceBoat(ocean, boat);

    // Boat #4
  boat.ID = 4;
  boat.position = {3, 8};
  boat.orientation = HLP2::WarBoats::oHORIZONTAL;
  HLP2::WarBoats::PlaceBoat(ocean, boat);

    // Boat #5
  boat.ID = 5;
  boat.position = {9, 6};
  boat.orientation = HLP2::WarBoats::oVERTICAL;
  HLP2::WarBoats::PlaceBoat(ocean, boat);

  std::cout << "The board with " << num_boats << " boats" << "\n";
  HLP2::WarBoats::DumpOcean(ocean);
  std::cout << "\n";

    // Try to sink the boats with random shots
  int shots {0};
  while (num_boats > 0) {
    HLP2::WarBoats::ShotResult sr;
    do {
      HLP2::WarBoats::Point coordinate;
      coordinate.x = HLP2::Utils::Random(0, xsize - 1);
      coordinate.y = HLP2::Utils::Random(0, ysize - 1);
      std::cout << "Shot #" << ++shots << " at coordinate " << coordinate.x << ", " << coordinate.y << "  ";

      sr = HLP2::WarBoats::TakeShot(ocean, coordinate);
      PrintShotResult(sr);
    } while (sr == HLP2::WarBoats::srDUPLICATE);

      // Sunk a boat
    num_boats = (sr == HLP2::WarBoats::srSUNK) ? num_boats-1 : num_boats;

    //if (sr == HLP2::WarBoats::srSUNK) {
      //PrintShotResult(sr);
      PrintStats(GetShotStats(ocean));
      HLP2::WarBoats::DumpOcean(ocean); std::cout << "\n";
    //}
  }
  HLP2::WarBoats::DestroyOcean(theOcean);
}

void test_rect(int num_boats, int xsize, int ysize, bool show_sunk, bool show_all) {
    // Initialize the pseudo-random number generator
  HLP2::Utils::srand(0, 0);

  HLP2::WarBoats::Ocean *theOcean = HLP2::WarBoats::CreateOcean(num_boats, xsize, ysize);
  HLP2::WarBoats::Ocean &ocean = *theOcean;

    // Place the boats randomly in the ocean
  int boats_placed {0};
  while (boats_placed < num_boats) {
    HLP2::WarBoats::Boat boat;

    boat.ID = boats_placed + 1;
    HLP2::WarBoats::BoatPlacement bp;
    do {
        // Pick a random orientation
      boat.orientation = HLP2::Utils::Random(0, 1) ? HLP2::WarBoats::oHORIZONTAL : HLP2::WarBoats::oVERTICAL;
      boat.position.x = HLP2::Utils::Random(0, xsize - 1); // Pick a random location
      boat.position.y = HLP2::Utils::Random(0, ysize - 1);
      bp = HLP2::WarBoats::PlaceBoat(ocean, boat); // Place the boat
    } while (bp == HLP2::WarBoats::bpREJECTED);
    ++boats_placed;
  }
  std::cout << "Boats placed:\n";
  HLP2::WarBoats::DumpOcean(ocean);
  std::cout << "\n";

    // Try to sink the boats with random shots
  while (num_boats > 0) {
    HLP2::WarBoats::ShotResult sr;
    do {
      HLP2::WarBoats::Point coordinate;
      coordinate.x = HLP2::Utils::Random(0, xsize - 1);
      coordinate.y = HLP2::Utils::Random(0, ysize - 1);
      //std::cout << "target: " << coordinate.x << ", "  << coordinate.y << "\n";

      sr = HLP2::WarBoats::TakeShot(ocean, coordinate);
    } while (sr == HLP2::WarBoats::srDUPLICATE);

      // Sunk a boat
    num_boats = (sr == HLP2::WarBoats::srSUNK) ? num_boats-1 : num_boats;
    if (show_all) {
      PrintShotResult(sr);
      PrintStats(GetShotStats(ocean));
      HLP2::WarBoats::DumpOcean(ocean, 4, false);
      std::cout << "\n";
    } else if ( (sr == HLP2::WarBoats::srSUNK) && show_sunk) {
      PrintShotResult(sr);
      PrintStats(GetShotStats(ocean));
      HLP2::WarBoats::DumpOcean(ocean, 4, false);
      std::cout << "\n";
    }
  }
  std::cout << "Final board:\n";
  PrintStats(GetShotStats(ocean));
  HLP2::WarBoats::DumpOcean(ocean, 4, false);
  std::cout << "\n";
  HLP2::WarBoats::DestroyOcean(theOcean);
}

} // end anonymous namespace
