////////////////////////////////////////////////////////////////////////////////
#ifndef OCEAN_H
#define OCEAN_H
////////////////////////////////////////////////////////////////////////////////

namespace HLP2 {
  namespace WarBoats {
    struct Ocean; //!< Forward declaration for the Ocean 

    enum Orientation   { oHORIZONTAL, oVERTICAL };
    enum ShotResult    { srHIT, srMISS, srDUPLICATE, srSUNK, srILLEGAL };
    enum DamageType    { dtOK = 0, dtBLOWNUP = -1 };
    enum BoatPlacement { bpACCEPTED, bpREJECTED };

      //! A coordinate in the Ocean
    struct Point {
      int x; //!< x-coordinate (column)
      int y; //!< y-coordinate (row)
    };

      //! A boat in the Ocean
    struct Boat {
      int hits;                 //!< Hits taken so far
      int ID;                   //!< Unique ID 
      Orientation orientation;  //!< Horizontal/Vertical
      Point position;           //!< x-y coordinate (left-top)
    };

      //! Statistics of the "game"
    struct ShotStats {
      int hits;       //!< The number of boat hits
      int misses;     //!< The number of boat misses
      int duplicates; //!< The number of duplicate (misses/hits)
      int sunk;       //!< The number of boats sunk
    };

      // The interface
    Ocean* CreateOcean(int num_boats, int x_size, int y_size);
    void DestroyOcean(Ocean *theOcean);
    ShotResult TakeShot(Ocean& ocean, Point const& coordinate);
    BoatPlacement PlaceBoat(Ocean& ocean, Boat const& boat);
    ShotStats GetShotStats(Ocean const& ocean);

      // Provided
    void DumpOcean(const Ocean &ocean, int width = 4, bool extraline = false, 
                   bool showboats = true);

  } // namespace WarBoats

} // namespace HLP2

namespace HLP2 {
  namespace WarBoats {
      //! The attributes of the ocean
    struct Ocean {
      int *grid;        //!< The 2D ocean 
      Boat *boats;      //!< The dynamic array of boats
      int num_boats;    //!< Number of boats in the ocean
      int x_size;       //!< Ocean size along x-axis
      int y_size;       //!< Ocean size along y-axis
      ShotStats stats;  //!< Status of the attack
    };
  } // namespace WarBoats
} // namespace HLP2

#endif // OCEAN_H
////////////////////////////////////////////////////////////////////////////////
