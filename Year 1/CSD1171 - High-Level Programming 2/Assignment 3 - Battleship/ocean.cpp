/**************************************************************************/
/*!
  \file ocean.cpp

  \author Bryan Ang Wei Ze

  \par DP email: bryanweize.ang\@digipen.edu

  \par Course: CSD1171 High-Level Programming 2

  \par Programming Assignment #3

  \date 01-24-2024

  \brief
    This source file implements functions to emulate a simple version of
    the popular board Battleship
    The functions include:
    - CreateOcean
    Creates and initializes an object of type Ocean using dynamically
    allocated memory.
    - DestroyOcean
    Deallocate all allocated memory for an object type Ocean
    - PlaceBoat
    Determines whether a boat can be placed
    - TakeShot
    Determines the result of the shot taken
    - GetShotStats
    Returns the statistics from the free store object referenced by ocean
*/
/**************************************************************************/
#include "ocean.h"
#include <iostream>
#include <iomanip>

namespace HLP2 {
  namespace WarBoats {
    int const BOAT_LENGTH {4};
    int const HIT_OFFSET  {100};

    /**************************************************************************/
    /*!
      \brief
        Creates and initializes an object of type Ocean using dynamically
        allocated memory
      
      \param num_boats
        Number of boats in the ocean

      \param x_size
        Ocean size along x-axis

      \param y_size
        Ocean size along y-axis

      \return
        A pointer to dynamically allocated object of type Ocean
      
    */
    /**************************************************************************/
    Ocean* CreateOcean(int num_boats, int x_size, int y_size)
    {
      Ocean* create
      {
        new Ocean
        {
            new int[x_size * y_size]{dtOK},
            new Boat[num_boats]{{0,0,oHORIZONTAL,{0,0}}},
            num_boats,
            x_size,
            y_size,
            {0,0,0,0}
        }
      };
      return create;
    }

    /**************************************************************************/
    /*!
      \brief
        Deallocates all allocated memory for an object type Ocean
      
      \param theOcean
        Pointer to the dynamically allocated object of type Ocean
      
    */
    /**************************************************************************/
    void DestroyOcean(Ocean *theOcean)
    {
      delete [] theOcean->boats;
      delete [] theOcean->grid;
      delete theOcean;
    }

    /**************************************************************************/
    /*!
      \brief
        Determines whether a boat can be placed
      
      \param ocean
        The Ocean to reference

      \param boat
        Reference to the boat to be placed

      \return
        The result of the placement
      
    */
    /**************************************************************************/
    BoatPlacement PlaceBoat(Ocean& ocean, Boat const& boat)
    {  
      // boat orientation is horizontal
      if (oHORIZONTAL == boat.orientation)
      {
        // any part of the boat is not within the ocean's dimensions
        if (boat.position.x + (BOAT_LENGTH-1) >= ocean.x_size ||boat.position.x < 0 || 
        boat.position.y < 0 || boat.position.y >= ocean.y_size)
        {
          return bpREJECTED;
        }
        // any part of the boat overlaps with another previously placed boat
        for (int i{0}; i < BOAT_LENGTH; i++)
        {
            if (ocean.grid[boat.position.y * ocean.x_size + (boat.position.x + i)] != dtOK)
            {
              return bpREJECTED;
            }
        }
        // place the boat horizontally
        for (int i{0}; i < BOAT_LENGTH; i++)
        {
          ocean.grid[boat.position.y * ocean.x_size + (boat.position.x + i)] = boat.ID;
       }
        
      }
      // boat orientation is vertical
      else
      {
        // any part of the boat is not within the ocean's dimensions
        if (boat.position.y + (BOAT_LENGTH-1) >= ocean.y_size || boat.position.y < 0 || 
        boat.position.x < 0 || boat.position.x >= ocean.x_size)
        {
          return bpREJECTED;
        }
        // any part of the boat overlaps with another previously placed boat
        for (int i{0}; i < BOAT_LENGTH; i++)
        {
            if (ocean.grid[(boat.position.y + i) * ocean.x_size + boat.position.x] != dtOK)
            {
              return bpREJECTED;
            }
        }
        // place the boat vertically
        for (int i{0}; i < BOAT_LENGTH; i++)
        {
          ocean.grid[(boat.position.y + i)  * ocean.x_size + boat.position.x] = boat.ID;
        }
      }

      return bpACCEPTED;
    }

    /**************************************************************************/
    /*!
      \brief
        Determines the result of the shot taken
      
      \param ocean
        The Ocean to reference

      \param coordinate
        The coordinates of the shot taken

      \return
        The result of the shot taken
      
    */
    /**************************************************************************/
    ShotResult TakeShot(Ocean& ocean, Point const& coordinate)
    {
      // shot hits outside the ocean dimensions
      if (coordinate.x < 0 || coordinate.x >= ocean.x_size || 
      coordinate.y < 0 || coordinate.y >= ocean.y_size)
      {
        return srILLEGAL;
      }
      
      // shot hits an open water position for the first time
      if (dtOK == ocean.grid[coordinate.y * ocean.x_size + coordinate.x])
      {
        ++ocean.stats.misses;
        ocean.grid[coordinate.y * ocean.x_size + coordinate.x] = dtBLOWNUP;
        return srMISS;
      }

      // shot re-hits either an open water position or a hit boat part
      if (dtBLOWNUP == ocean.grid[coordinate.y * ocean.x_size + coordinate.x] ||
      ocean.grid[coordinate.y * ocean.x_size + coordinate.x] >= HIT_OFFSET)
      {
        ++ocean.stats.duplicates;
        return srDUPLICATE;
      }

      // shot hits an un-hit part of a boat
      if (ocean.grid[coordinate.y * ocean.x_size + coordinate.x] > 0 &&
      ocean.grid[coordinate.y * ocean.x_size + coordinate.x] < HIT_OFFSET)
      {
        // save ID of boat that is hit
        int id_of_boat_hit{ocean.grid[coordinate.y * ocean.x_size + coordinate.x]};
        ++ocean.stats.hits;
        ++ocean.boats[id_of_boat_hit-1].hits;
        ocean.grid[coordinate.y * ocean.x_size + coordinate.x] += HIT_OFFSET;
        
        // a boat sinks after all 4 parts are hit
        if (BOAT_LENGTH == ocean.boats[id_of_boat_hit-1].hits)
        {
          ++ocean.stats.sunk;
          return srSUNK;
        } 
      }
      return srHIT;
    }

    /**************************************************************************/
    /*!
      \brief
        Returns the statistics from the free store object referenced by ocean
      
      \param ocean
        The Ocean to reference

      \return
        The statistics from the free store object referenced by ocean
      
    */
    /**************************************************************************/
    ShotStats GetShotStats(Ocean const& ocean)
    {
      return ocean.stats;
    }

    /**************************************************************************/
    /*!
      \brief
        Prints the grid (ocean) to the screen.
      
      \param ocean
        The Ocean to print
      
      \param field_width
        How much space each position takes when printed.
      
      \param extraline
        If true, an extra line is printed after each row. If false, no extra
        line is printed.
        
      \param showboats
        If true, the boats are shown in the output. (Debugging feature)
    */
    /**************************************************************************/
    void DumpOcean(const HLP2::WarBoats::Ocean &ocean,
                                    int field_width, 
                                    bool extraline, 
                                    bool showboats) {
      for (int y{0}; y < ocean.y_size; y++) { // For each row
        for (int x{0}; x < ocean.x_size; x++) { // For each column
            // Get value at x/y position
          int value = ocean.grid[y * ocean.x_size + x];
            // Is it a boat that we need to keep hidden?
          value = ( (value > 0) && (value < HIT_OFFSET) && (showboats == false) ) ? 0 : value;
          std::cout << std::setw(field_width) << value;
        }
        std::cout << "\n";
        if (extraline) { std::cout << "\n"; }
      }
    }
  } // namespace WarBoats
} // namespace HLP2
