/**************************************************************************/
/*!
  \file ocean.h

  \author Bryan Ang Wei Ze

  \par DP email: bryanweize.ang\@digipen.edu

  \par Course: CSD1171 High-Level Programming 2

  \par Programming Assignment #5

  \date 02-09-2024

  \brief
    This header file declares functions to emulate a simple version of
    the popular board Battleship using classes
*/
/**************************************************************************/
#ifndef OCEAN_H
#define OCEAN_H

namespace HLP2 {
  namespace WarBoats {
    inline int const BOAT_LENGTH {4};
    inline int const HIT_OFFSET  {100};
    class Ocean; //!< Forward declaration for the Ocean 

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

  } // namespace WarBoats
} // namespace HLP2

namespace HLP2 {
  namespace WarBoats {
      //! The attributes of the ocean
    class Ocean {
      public:
      
      /**************************************************************************/
        /*!
        \brief
        Creates and initializes an object of type Ocean
      
        \param n_boats
        Number of boats in the ocean

        \param x
        Ocean size along x-axis

        \param y
        Ocean size along y-axis
        */
      /**************************************************************************/
      Ocean(int n_boats, int x, int y);
      
      /**************************************************************************/
        /*!
        \brief
            Deallocates all allocated memory for an object type Ocean
        */
      /**************************************************************************/
      ~Ocean();
      
      /**************************************************************************/
        /*!
        \brief
            Determines whether a boat can be placed

        \param boat
            Reference to the boat to be placed

        \return
            The result of the placement
        */
      /**************************************************************************/
      BoatPlacement PlaceBoat(Boat const& boat) const;
      
      /**************************************************************************/
        /*!
        \brief
            Determines the result of the shot taken

        \param coordinate
            The coordinates of the shot taken

        \return
            The result of the shot taken
        */
      /**************************************************************************/
      ShotResult TakeShot(Point const& coordinate);
      
      /**************************************************************************/
        /*!
        \brief
            Returns the statistics from the free store object referenced by ocean

        \return
            The statistics from the free store object referenced by ocean
        */
      /**************************************************************************/
      ShotStats GetShotStats() const;
      
      /**************************************************************************/
        /*!
        \brief
            Returns the grid of the ocean

        \return
            A pointer to dynamically allocated ocean grid
        */
      /**************************************************************************/
      int* GetGrid() const;
      
      /**************************************************************************/
        /*!
        \brief
            Returns the dimensions of the ocean

        \return
            The x and y sizes of the ocean
        */
      /**************************************************************************/
      Point GetDimensions() const;

      private:
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
