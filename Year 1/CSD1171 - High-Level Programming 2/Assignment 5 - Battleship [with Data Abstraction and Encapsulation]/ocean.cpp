/**************************************************************************/
/*!
  \file ocean.cpp

  \author Bryan Ang Wei Ze

  \par DP email: bryanweize.ang\@digipen.edu

  \par Course: CSD1171 High-Level Programming 2

  \par Programming Assignment #5

  \date 02-09-2024

  \brief
    This source file implements functions to emulate a simple version of
    the popular board Battleship using classes
*/
/**************************************************************************/
#include "ocean.h"
#include <iostream>
#include <iomanip>

namespace HLP2 {
  namespace WarBoats {

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
    Ocean::Ocean(int n_boats, int x, int y)
    : grid{new int[x * y]{dtOK}}, boats{new Boat[n_boats]{{0,0,oHORIZONTAL,{0,0}}}},
    num_boats{n_boats}, x_size{x}, y_size{y}, stats{0,0,0,0}
    {

    }

    /**************************************************************************/
    /*!
      \brief
        Deallocates all allocated memory for an object type Ocean
    */
    /**************************************************************************/
    Ocean::~Ocean()
    {
      delete[] boats;
      delete[] grid;
    }
    
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
    BoatPlacement Ocean::PlaceBoat(Boat const& boat) const
    {
      // boat orientation is horizontal
      if (oHORIZONTAL == boat.orientation)
      {
        // any part of the boat is not within the ocean's dimensions
        if (boat.position.x + (BOAT_LENGTH-1) >= x_size ||  boat.position.x < 0 || 
        boat.position.y < 0 || boat.position.y >= y_size)
        {
          return bpREJECTED;
        }
        // any part of the boat overlaps with another previously placed boat
        for (int i = 0; i < BOAT_LENGTH; i++)
        {
            if (grid[boat.position.y * x_size + (boat.position.x + i)] != dtOK)
            {
              return bpREJECTED;
            }
        }
        // place the boat horizontally
        for (int i = 0; i < BOAT_LENGTH; i++)
        {
          grid[boat.position.y * x_size + (boat.position.x + i)] = boat.ID;
        }
      }
      else
      {
        // any part of the boat is not within the ocean's dimensions
        if (boat.position.y + (BOAT_LENGTH-1) >= y_size || boat.position.y < 0 || 
        boat.position.x < 0 || boat.position.x >= x_size)
        {
          return bpREJECTED;
        }
        // any part of the boat overlaps with another previously placed boat
        for (int i = 0; i < BOAT_LENGTH; i++)
        {
            if (grid[(boat.position.y + i) * x_size + boat.position.x] != dtOK)
            {
              return bpREJECTED;
            }
        }
        // place the boat vertically
        for (int i = 0; i < BOAT_LENGTH; i++)
        {
          grid[(boat.position.y + i)  * x_size + boat.position.x] = boat.ID;
        }
      }
    
     return bpACCEPTED;
    }

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
    ShotResult Ocean::TakeShot(Point const& coordinate)
    {
      // shot hits outside the ocean dimensions
      if (coordinate.x < 0 || coordinate.x >= x_size || 
      coordinate.y < 0 || coordinate.y >= y_size)
      {
        return srILLEGAL;
      }
      
      // shot hits an open water position for the first time
      if (dtOK == grid[coordinate.y * x_size + coordinate.x])
      {
        ++stats.misses;
        grid[coordinate.y * x_size + coordinate.x] = dtBLOWNUP;
        return srMISS;
      }

      // shot re-hits either an open water position or a hit boat part
      if (dtBLOWNUP == grid[coordinate.y * x_size + coordinate.x] ||
      grid[coordinate.y * x_size + coordinate.x] >= HIT_OFFSET)
      {
        ++stats.duplicates;
        return srDUPLICATE;
      }

      // shot hits an un-hit part of a boat
      if (grid[coordinate.y * x_size + coordinate.x] > 0 &&
      grid[coordinate.y * x_size + coordinate.x] < HIT_OFFSET)
      {
        // save ID of boat that is hit
        int id_of_boat_hit{grid[coordinate.y * x_size + coordinate.x]};
        ++stats.hits;
        ++boats[id_of_boat_hit-1].hits;
        grid[coordinate.y * x_size + coordinate.x] += HIT_OFFSET;
        
        // a boat sinks after all 4 parts are hit
        if (BOAT_LENGTH == boats[id_of_boat_hit-1].hits)
        {
          ++stats.sunk;
          return srSUNK;
        } 
      }
      return srHIT;
    }

    /**************************************************************************/
    /*!
      \brief
        Returns the statistics from the free store object referenced by ocean

      \return
        The statistics from the free store object referenced by ocean
    */
    /**************************************************************************/
    ShotStats Ocean::GetShotStats() const
    {
      return stats;
    }
    
    /**************************************************************************/
    /*!
      \brief
        Returns the grid of the ocean

      \return
        A pointer to dynamically allocated ocean grid
    */
    /**************************************************************************/
    int* Ocean::GetGrid() const
    {
      return grid;
    }
    
    /**************************************************************************/
    /*!
      \brief
        Returns the dimensions of the ocean

      \return
        The x and y sizes of the ocean
    */
    /**************************************************************************/
    Point Ocean::GetDimensions() const
    {
      Point result{x_size, y_size};
      return result;
    }
  } // namespace WarBoats
} // namespace HLP2
