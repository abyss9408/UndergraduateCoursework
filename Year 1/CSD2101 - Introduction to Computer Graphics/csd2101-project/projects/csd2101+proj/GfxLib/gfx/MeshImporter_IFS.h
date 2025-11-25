/*!
@file    MeshImporter_IFS.h
@author  Prasanna Ghali       (pghali@digipen.edu)

CVS: $Id: MeshImporter_IFS.h,v 1.2 2005/02/16 21:15:41 jmp Exp $

All content (c) 2002 DigiPen Institute of Technology, all rights reserved.
*//*__________________________________________________________________________*/

/*                                                                      guard
----------------------------------------------------------------------------- */

#ifndef GFX_MESHIMPORTER_IFS_H_
#define GFX_MESHIMPORTER_IFS_H_

/*                                                                   includes
----------------------------------------------------------------------------- */

#include "GFXInternal.h"

/*                                                                    classes
----------------------------------------------------------------------------- */

// forward declarations
class gfxModel;

/*  _________________________________________________________________________ */
class gfxMeshImporter_IFS
/*! IFS mesh importer.

    IFS (Indexed Face Format) is a simple binary format that stores only vertex
    and index information. The models in the Brown Mesh Set, available from
    
    http://graphics.cs.brown.edu/games/brown-mesh-set/
    
    are stored in IFS format. IFS only stores vertex and index information,
    however, the importer can general vertex normals if desired.
*/
{ 
  public:
    // importer
    static gfxModel Import(const std::string& fn, bool generate_normals = false);
  
  private:
    // disabled
     gfxMeshImporter_IFS(); 
    ~gfxMeshImporter_IFS();
    
    // helpers
    static std::string  nReadString32(std::ifstream& ifs);
    static float        nReadFloat32(std::ifstream& ifs);
    static unsigned int nReadInt32(std::ifstream& ifs);

    // data members
    static char*				smStringReadBuffer;    //!< Buffer for reading strings.
    static unsigned int	smStringReadBufferSz;  //!< Current buffer size.
};


#endif  /* GFX_MESHIMPORTER_IFS_H_ */