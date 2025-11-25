/******************************************************************************/
/*!
\file		GameState_Platform.cpp
\author 	Bryan Ang Wei Ze
\par    	email: bryanweize.ang\@digipen.edu
\date   	February 28, 2024
\brief		This source file implements the Load, Initialize, Update, Draw, Free
			and Unload functions of GameState_Platform.

Copyright (C) 2024 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
 */
/******************************************************************************/

#include "main.h"
#include <iostream>

/******************************************************************************/
/*!
	Defines
*/
/******************************************************************************/
const unsigned int	GAME_OBJ_NUM_MAX		= 32;	//The total number of different objects (Shapes)
const unsigned int	GAME_OBJ_INST_NUM_MAX	= 2048;	//The total number of different game object instances

//Gameplay related variables and values
const float			GRAVITY					= -20.0f;
const float			JUMP_VELOCITY			= 11.0f;
const float			MOVE_VELOCITY_HERO		= 4.0f;
const float			MOVE_VELOCITY_ENEMY		= 7.5f;
const double		ENEMY_IDLE_TIME			= 2.0;
const int			HERO_LIVES				= 3;

//Particle related variables and values
const double		PARTICLE_MIN_LIEFTIME	  = 0.5f;
const double		PARTICLE_MAX_LIEFTIME	  = 1.0f;
const float			PARTICLE_MIN_VELOCITY_X	  = -1.5f;
const float			PARTICLE_MAX_VELOCITY_X	  = 1.5f;
const float			PARTICLE_MIN_VELOCITY_Y	  = 3.5f;
const float			PARTICLE_MAX_VELOCITY_Y   = 7.0f;
const int			MAX_PARTICLES			  = 35;

//Flags
const unsigned int	FLAG_ACTIVE				= 0x00000001;
const unsigned int	FLAG_VISIBLE			= 0x00000002;
const unsigned int	FLAG_NON_COLLIDABLE		= 0x00000004;

//Collision flags
const unsigned int	COLLISION_LEFT			= 0x00000001;	//0001
const unsigned int	COLLISION_RIGHT			= 0x00000002;	//0010
const unsigned int	COLLISION_TOP			= 0x00000004;	//0100
const unsigned int	COLLISION_BOTTOM		= 0x00000008;	//1000


enum TYPE_OBJECT
{
	TYPE_OBJECT_EMPTY,			//0
	TYPE_OBJECT_COLLISION,		//1
	TYPE_OBJECT_HERO,			//2
	TYPE_OBJECT_ENEMY1,			//3
	TYPE_OBJECT_COIN,			//4
	TYPE_OBJECT_PARTICLE		//5
};

//State machine states
enum STATE
{
	STATE_NONE,
	STATE_GOING_LEFT,
	STATE_GOING_RIGHT
};

//State machine inner states
enum INNER_STATE
{
	INNER_STATE_ON_ENTER,
	INNER_STATE_ON_UPDATE,
	INNER_STATE_ON_EXIT
};

/******************************************************************************/
/*!
	Struct/Class Definitions
*/
/******************************************************************************/
struct GameObj
{
	unsigned int		type;		// object type
	AEGfxVertexList *	pMesh;		// pbject
};


struct GameObjInst
{
	GameObj *		pObject;	// pointer to the 'original'
	unsigned int	flag;		// bit flag or-ed together
	AEVec2			scale;		// scaling value of the object instance
	AEVec2			posCurr;	// object current position
	AEVec2			velCurr;	// object current velocity
	float			dirCurr;	// object current direction

	AEVec2			posPrev;	// object previous position -> it's the position calculated in the previous loop

	AEMtx33			transform;	// object drawing matrix
	
	AABB			boundingBox;// object bouding box that encapsulates the object

	//Used to hold the current 
	int				gridCollisionFlag;

	// pointer to custom data specific for each object type
	void*			pUserData;

	//State of the object instance
	enum			STATE state;
	enum			INNER_STATE innerState;

	//General purpose counter (This variable will be used for the enemy state machine)
	double			counter;
};


/******************************************************************************/
/*!
	File globals
*/
/******************************************************************************/
static int				HeroLives;
static int				Hero_Initial_X;
static int				Hero_Initial_Y;
static int				TotalCoins;
static int				NumParticles;

// list of original objects
static GameObj			*sGameObjList;
static unsigned int		sGameObjNum;

// list of object instances
static GameObjInst		*sGameObjInstList;
static unsigned int		sGameObjInstNum;

//Binary map data
static int				**MapData;
static int				**BinaryCollisionArray;
static int				BINARY_MAP_WIDTH;
static int				BINARY_MAP_HEIGHT;
static GameObjInst		*pBlackInstance;
static GameObjInst		*pWhiteInstance;
static AEMtx33			MapTransform;
static AEVec2			MapScale, MapTranslate;

int						GetCellValue(int X, int Y);
int						CheckInstanceBinaryMapCollision(float PosX, float PosY, 
														float scaleX, float scaleY);
void					SnapToCell(float *Coordinate);
int						ImportMapDataFromFile(char *FileName);
void					FreeMapData(void);

// function to create/destroy a game object instance
static GameObjInst*		gameObjInstCreate (unsigned int type, AEVec2* scale,
											AEVec2* pPos, AEVec2* pVel, 
											float dir, enum STATE startState);
static void				gameObjInstDestroy(GameObjInst* pInst);

//We need a pointer to the hero's instance for input purposes
static GameObjInst		*pHero;

static GameObjInst		*pParticle[MAX_PARTICLES];

//State machine functions
void					EnemyStateMachine(GameObjInst *pInst);


/******************************************************************************/
/*!
	"Load" function of this state
*/
/******************************************************************************/
void GameStatePlatformLoad(void)
{
	sGameObjList = (GameObj *)calloc(GAME_OBJ_NUM_MAX, sizeof(GameObj));
	sGameObjInstList = (GameObjInst *)calloc(GAME_OBJ_INST_NUM_MAX, sizeof(GameObjInst));
	sGameObjNum = 0;


	GameObj* pObj;

	//Creating the black object
	pObj = sGameObjList + sGameObjNum++;

	if (pObj)
	{
		pObj->type = TYPE_OBJECT_EMPTY;


		AEGfxMeshStart();
		AEGfxTriAdd(
			-0.5f, -0.5f, 0xFF000000, 0.0f, 0.0f,
			0.5f, -0.5f, 0xFF000000, 0.0f, 0.0f,
			-0.5f, 0.5f, 0xFF000000, 0.0f, 0.0f);

		AEGfxTriAdd(
			-0.5f, 0.5f, 0xFF000000, 0.0f, 0.0f,
			0.5f, -0.5f, 0xFF000000, 0.0f, 0.0f,
			0.5f, 0.5f, 0xFF000000, 0.0f, 0.0f);

		pObj->pMesh = AEGfxMeshEnd();
		AE_ASSERT_MESG(pObj->pMesh, "fail to create object!!");


		//Creating the white object
		pObj = sGameObjList + sGameObjNum++;
		pObj->type = TYPE_OBJECT_COLLISION;


		AEGfxMeshStart();
		AEGfxTriAdd(
			-0.5f, -0.5f, 0xFFFFFFFF, 0.0f, 0.0f,
			0.5f, -0.5f, 0xFFFFFFFF, 0.0f, 0.0f,
			-0.5f, 0.5f, 0xFFFFFFFF, 0.0f, 0.0f);

		AEGfxTriAdd(
			-0.5f, 0.5f, 0xFFFFFFFF, 0.0f, 0.0f,
			0.5f, -0.5f, 0xFFFFFFFF, 0.0f, 0.0f,
			0.5f, 0.5f, 0xFFFFFFFF, 0.0f, 0.0f);

		pObj->pMesh = AEGfxMeshEnd();
		AE_ASSERT_MESG(pObj->pMesh, "fail to create object!!");


		//Creating the hero object
		pObj = sGameObjList + sGameObjNum++;
		pObj->type = TYPE_OBJECT_HERO;


		AEGfxMeshStart();
		AEGfxTriAdd(
			-0.5f, -0.5f, 0xFF0000FF, 0.0f, 0.0f,
			0.5f, -0.5f, 0xFF0000FF, 0.0f, 0.0f,
			-0.5f, 0.5f, 0xFF0000FF, 0.0f, 0.0f);

		AEGfxTriAdd(
			-0.5f, 0.5f, 0xFF0000FF, 0.0f, 0.0f,
			0.5f, -0.5f, 0xFF0000FF, 0.0f, 0.0f,
			0.5f, 0.5f, 0xFF0000FF, 0.0f, 0.0f);

		pObj->pMesh = AEGfxMeshEnd();
		AE_ASSERT_MESG(pObj->pMesh, "fail to create object!!");


		//Creating the enemey1 object
		pObj = sGameObjList + sGameObjNum++;
		pObj->type = TYPE_OBJECT_ENEMY1;


		AEGfxMeshStart();
		AEGfxTriAdd(
			-0.5f, -0.5f, 0xFFFF0000, 0.0f, 0.0f,
			0.5f, -0.5f, 0xFFFF0000, 0.0f, 0.0f,
			-0.5f, 0.5f, 0xFFFF0000, 0.0f, 0.0f);

		AEGfxTriAdd(
			-0.5f, 0.5f, 0xFFFF0000, 0.0f, 0.0f,
			0.5f, -0.5f, 0xFFFF0000, 0.0f, 0.0f,
			0.5f, 0.5f, 0xFFFF0000, 0.0f, 0.0f);

		pObj->pMesh = AEGfxMeshEnd();
		AE_ASSERT_MESG(pObj->pMesh, "fail to create object!!");


		//Creating the Coin object
		pObj = sGameObjList + sGameObjNum++;
		pObj->type = TYPE_OBJECT_COIN;


		AEGfxMeshStart();
		//Creating the circle shape
		int Parts = 12;
		for (float i = 0; i < Parts; ++i)
		{
			AEGfxTriAdd(
				0.0f, 0.0f, 0xFFFFFF00, 0.0f, 0.0f,
				cosf(i * 2 * PI / Parts) * 0.5f, sinf(i * 2 * PI / Parts) * 0.5f, 0xFFFFFF00, 0.0f, 0.0f,
				cosf((i + 1) * 2 * PI / Parts) * 0.5f, sinf((i + 1) * 2 * PI / Parts) * 0.5f, 0xFFFFFF00, 0.0f, 0.0f);
		}

		pObj->pMesh = AEGfxMeshEnd();
		AE_ASSERT_MESG(pObj->pMesh, "fail to create object!!");


		//Creating the Particle object
		pObj = sGameObjList + sGameObjNum++;
		pObj->type = TYPE_OBJECT_PARTICLE;


		AEGfxMeshStart();
		AEGfxTriAdd(
			-0.5f, -0.5f, 0xA09900CC, 0.0f, 0.0f,
			0.5f, -0.5f, 0xA09900CC, 0.0f, 0.0f,
			-0.5f, 0.5f, 0xA09900CC, 0.0f, 0.0f);

		AEGfxTriAdd(
			-0.5f, 0.5f, 0xA09900CC, 0.0f, 0.0f,
			0.5f, -0.5f, 0xA09900CC, 0.0f, 0.0f,
			0.5f, 0.5f, 0xA09900CC, 0.0f, 0.0f);

		pObj->pMesh = AEGfxMeshEnd();
		AE_ASSERT_MESG(pObj->pMesh, "fail to create object!!");
	}
}

/******************************************************************************/
/*!
	"Initialize" function of this state
*/
/******************************************************************************/
void GameStatePlatformInit(void)
{
	int i{}, j{};
	//UNREFERENCED_PARAMETER(j);

	//Setting intital binary map values
	MapData = 0;
	BinaryCollisionArray = 0;
	BINARY_MAP_WIDTH = 0;
	BINARY_MAP_HEIGHT = 0;

	pHero = 0;
	pBlackInstance = 0;
	pWhiteInstance = 0;
	TotalCoins = 0;
	NumParticles = 0;

	//Create an object instance representing the black cell.
	//This object instance should not be visible. When rendering the grid cells, each time we have
	//a non collision cell, we position this instance in the correct location and then we render it
	AEVec2 scl = { 1.0f, 1.0f };
	pBlackInstance = gameObjInstCreate(TYPE_OBJECT_EMPTY, &scl, 0, 0, 0.0f, STATE_NONE);
	pBlackInstance->flag ^= FLAG_VISIBLE;
	pBlackInstance->flag |= FLAG_NON_COLLIDABLE;

	//Create an object instance representing the white cell.
	//This object instance should not be visible. When rendering the grid cells, each time we have
	//a collision cell, we position this instance in the correct location and then we render it
	pWhiteInstance = gameObjInstCreate(TYPE_OBJECT_COLLISION, &scl, 0, 0, 0.0f, STATE_NONE);
	pWhiteInstance->flag ^= FLAG_VISIBLE;
	pWhiteInstance->flag |= FLAG_NON_COLLIDABLE;

	//Setting the inital number of hero lives
	HeroLives = HERO_LIVES;

	//Entering level 1
	if (level == 1)
	{
		//Importing level 1 Data
		if (!ImportMapDataFromFile("../Resources/Levels/Exported.txt"))
			gGameStateNext = GS_QUIT;
	}
	//Entering level 2
	else
	{
		//Importing level 2 Data
		if (!ImportMapDataFromFile("../Resources/Levels/Exported2.txt"))
			gGameStateNext = GS_QUIT;
	}
	
	//Computing the matrix which take a point out of the normalized coordinates system
	//of the binary map
	/***********
	Compute a transformation matrix and save it in "MapTransform".
	This transformation transforms any point from the normalized coordinates system of the binary map.
	Later on, when rendering each object instance, we should concatenate "MapTransform" with the
	object instance's own transformation matrix

	Compute a translation matrix (-(Grid width/2), -Grid height/2) and save it in "trans"
	Compute a scaling matrix and save it in "scale". The scale must account for the window width and height.
		Alpha engine has 2 helper functions to get the window width and height: AEGetWindowWidth() and AEGetWindowHeight()
	Concatenate scale and translate and save the result in "MapTransform"
	***********/
	AEMtx33 scale{ 0 }, trans{ 0 };


	if (level == 1)
	{
		MapScale.x = static_cast<f32>(AEGfxGetWindowWidth()) / BINARY_MAP_WIDTH;
		MapScale.y = static_cast<f32>(AEGfxGetWindowHeight()) / BINARY_MAP_HEIGHT;
	}
	else
	{
		MapScale.x = static_cast<f32>(AEGfxGetWindowWidth()) / BINARY_MAP_WIDTH * 2;
		MapScale.y = static_cast<f32>(AEGfxGetWindowHeight()) / BINARY_MAP_HEIGHT * 2;
	}

	MapTranslate.x = -static_cast<f32>(AEGfxGetWindowWidth()) / 2 + MapScale.x / 2;
	MapTranslate.y = -static_cast<f32>(AEGfxGetWindowHeight()) / 2 + MapScale.y / 2;

	AEMtx33Scale(&scale, MapScale.x, MapScale.y);
	AEMtx33Trans(&trans, MapTranslate.x, MapTranslate.y);
	AEMtx33Concat(&MapTransform, &trans, &scale);

	GameObjInst *pInst = 0;
	AEVec2 Pos{ 0.0f,0.0f };

	UNREFERENCED_PARAMETER(pInst);

	// creating the main character, the enemies and the coins according 
	// to their initial positions in MapData

	/***********
	Loop through all the array elements of MapData 
	(which was initialized in the "GameStatePlatformLoad" function
	from the .txt file
		if the element represents a collidable or non collidable area
			don't do anything

		if the element represents the hero
			Create a hero instance
			Set its position depending on its array indices in MapData
			Save its array indices in Hero_Initial_X and Hero_Initial_Y 
			(Used when the hero dies and its position needs to be reset)

		if the element represents an enemy
			Create an enemy instance
			Set its position depending on its array indices in MapData
			
		if the element represents a coin
			Create a coin instance
			Set its position depending on its array indices in MapData
			
	***********/
	for (i = 0; i < BINARY_MAP_WIDTH; ++i)
	{
		for (j = 0; j < BINARY_MAP_HEIGHT; ++j)
		{
			Pos.x = static_cast<f32>(i) + 0.5f;
			Pos.y = static_cast<f32>(j) + 0.5f;
			switch (MapData[i][j])
			{
			case TYPE_OBJECT_EMPTY:
				break;
			case TYPE_OBJECT_COLLISION:
				break;
			case TYPE_OBJECT_HERO:
				Hero_Initial_X = i;
				Hero_Initial_Y = j;
				pHero = gameObjInstCreate(TYPE_OBJECT_HERO, &scl, &Pos, 0, 0.0f, STATE_NONE);
				break;
			case TYPE_OBJECT_ENEMY1:
				gameObjInstCreate(TYPE_OBJECT_ENEMY1, &scl, &Pos, 0, 0.0f, STATE_GOING_LEFT);
				break;
			case TYPE_OBJECT_COIN:
				gameObjInstCreate(TYPE_OBJECT_COIN, &scl, &Pos, 0, 0.0f, STATE_NONE);
				++TotalCoins;
				break;
			}
		}
	}

	AEVec2 random_vel{ 0.f, 0.f };
	AEVec2 particle_pos = { static_cast<f32>(Hero_Initial_X), static_cast<f32>(Hero_Initial_Y) };
	AEVec2 particle_scl = { 0.4f, 0.4f };
	double lifetime;

	for (i = 0; i < MAX_PARTICLES; ++i)
	{
		lifetime = PARTICLE_MIN_LIEFTIME + static_cast <double> (rand()) / (static_cast <double> (RAND_MAX / (PARTICLE_MAX_LIEFTIME - PARTICLE_MIN_LIEFTIME)));
		random_vel.x = PARTICLE_MIN_VELOCITY_X + static_cast <f32> (rand()) / (static_cast <f32> (RAND_MAX / (PARTICLE_MAX_VELOCITY_X - PARTICLE_MIN_VELOCITY_X)));
		random_vel.y = PARTICLE_MIN_VELOCITY_Y + static_cast <f32> (rand()) / (static_cast <f32> (RAND_MAX / (PARTICLE_MAX_VELOCITY_Y - PARTICLE_MIN_VELOCITY_Y)));
		pParticle[i] = gameObjInstCreate(TYPE_OBJECT_PARTICLE, &particle_scl, &particle_pos, &random_vel, 0.0f, STATE_NONE);
		pParticle[i]->counter = lifetime;
	}
	

	// reset camera position
	AEGfxSetCamPosition(0.f, 0.f);
}

/******************************************************************************/
/*!
	"Update" function of this state
*/
/******************************************************************************/
void GameStatePlatformUpdate(void)
{
	int i{}, j{};
	GameObjInst *pInst = 0;
	
	UNREFERENCED_PARAMETER(j);
	UNREFERENCED_PARAMETER(pInst);
	//Handle Input
	/***********
	if right is pressed
		Set hero velocity X to MOVE_VELOCITY_HERO
	else
	if left is pressed
		Set hero velocity X to -MOVE_VELOCITY_HERO
	else
		Set hero velocity X to 0

	if space is pressed AND Hero is colliding from the bottom
		Set hero velocity Y to JUMP_VELOCITY

	if Escape is pressed
		Exit to menu
	***********/
	if (AEInputCheckCurr(AEVK_RIGHT))
	{
		pHero->velCurr.x = MOVE_VELOCITY_HERO;
		pHero->dirCurr = 0.0f;
	}
	else if (AEInputCheckCurr(AEVK_LEFT))
	{
		pHero->velCurr.x = -MOVE_VELOCITY_HERO;
		pHero->dirCurr = PI;
	}
	else
	{
		pHero->velCurr.x = 0.f;
	}

	// level 1 jumping mechanic
	if (level == 1)
	{
		if (AEInputCheckCurr(AEVK_SPACE) && pHero->gridCollisionFlag & COLLISION_BOTTOM)
		{
			pHero->velCurr.y = JUMP_VELOCITY;
		}
	}
	// level 2 jumping mechanic
	else
	{
		// player can wall jump by colliding with wall and triggering the opposite direction key
		// example: player can jump when he/she collides with left wall and triggered move right key,
		if (AEInputCheckCurr(AEVK_SPACE) && (pHero->gridCollisionFlag & COLLISION_BOTTOM || 
			(pHero->gridCollisionFlag & COLLISION_LEFT && AEInputCheckTriggered(AEVK_RIGHT)) || 
			pHero->gridCollisionFlag & COLLISION_RIGHT && AEInputCheckTriggered(AEVK_LEFT)))
		{
			pHero->velCurr.y = JUMP_VELOCITY;
		}
	}

	if (AEInputCheckCurr(AEVK_ESCAPE))
	{
		gGameStateNext = GS_MENU;
	}

	//Update object instances physics and behavior
	for(i = 0; i < GAME_OBJ_INST_NUM_MAX; ++i)
	{
		pInst = sGameObjInstList + i;

		// skip non-active object
		if ((pInst->flag & FLAG_ACTIVE) == 0)
			continue;


		/****************
		Apply gravity
			Velocity Y = Gravity * Frame Time + Velocity Y

		If object instance is an enemy
			Apply enemy state machine
		****************/
		if (pInst->pObject->type == TYPE_OBJECT_HERO || pInst->pObject->type == TYPE_OBJECT_ENEMY1)
		{
			pInst->velCurr.y = GRAVITY * g_dt + pInst->velCurr.y;
		}

		if (pInst->pObject->type == TYPE_OBJECT_ENEMY1)
		{
			EnemyStateMachine(pInst);
		}

		if (pInst->pObject->type == TYPE_OBJECT_PARTICLE)
		{
			pInst->counter -= g_dt;
			pInst->scale.x -= g_dt / 3;
			pInst->scale.y -= g_dt / 3;
			if (pInst->counter <= 0.0)
			{
				gameObjInstDestroy(pInst);

				AEVec2 random_vel{ 0.0f,0.0f };
				AEVec2 particle_scl{ 0.4f, 0.4f };
				AEVec2 particle_pos{ 0.0f,0.0f };
				double lifetime{ 0.0f };

				if (pHero->dirCurr == 0.0f)
				{
					particle_pos.x = pHero->posCurr.x - 0.5f;
					particle_pos.y = pHero->posCurr.y;
				}
				else
				{
					particle_pos.x = pHero->posCurr.x;
					particle_pos.y = pHero->posCurr.y;
				}

				lifetime = PARTICLE_MIN_LIEFTIME + static_cast <double> (rand()) / (static_cast <double> (RAND_MAX / (PARTICLE_MAX_LIEFTIME - PARTICLE_MIN_LIEFTIME)));
				random_vel.x = PARTICLE_MIN_VELOCITY_X + static_cast <f32> (rand()) / (static_cast <f32> (RAND_MAX / (PARTICLE_MAX_VELOCITY_X - PARTICLE_MIN_VELOCITY_X)));
				random_vel.y = PARTICLE_MIN_VELOCITY_Y + static_cast <f32> (rand()) / (static_cast <f32> (RAND_MAX / (PARTICLE_MAX_VELOCITY_Y - PARTICLE_MIN_VELOCITY_Y)));

				pInst = gameObjInstCreate(TYPE_OBJECT_PARTICLE, &particle_scl, &particle_pos, &random_vel, 0.0f, STATE_NONE);
				pInst->counter = lifetime;
			}
		}
	}



	// ======================================================
	// Save previous positions
	//  -- For all instances
	// ======================================================
	for (i = 0; i < GAME_OBJ_INST_NUM_MAX; i++)
	{
		pInst = sGameObjInstList + i;

		// skip non-active object
		if ((pInst->flag & FLAG_ACTIVE) == 0)
			continue;

		pInst->posPrev.x = pInst->posCurr.x;
		pInst->posPrev.y = pInst->posCurr.y;
	}



	//Update object instances positions
	for(i = 0; i < GAME_OBJ_INST_NUM_MAX; ++i)
	{
		pInst = sGameObjInstList + i;

		// skip non-active object
		if ((pInst->flag & FLAG_ACTIVE) == 0)
			continue;

		/**********
		Get the bounding rectangle of every active instance:
			boundingRect_min = -BOUNDING_RECT_SIZE * instance->scale + instance->pos
			boundingRect_max = BOUNDING_RECT_SIZE * instance->scale + instance->pos

		Update the position using: P1 = V1*dt + P0
		**********/

		pInst->boundingBox.min.x = -1.0f / 2.0f * pInst->scale.x + pInst->posCurr.x;
		pInst->boundingBox.min.y = -1.0f / 2.0f * pInst->scale.y + pInst->posCurr.y;

		pInst->boundingBox.max.x = 1.0f / 2.0f * pInst->scale.x + pInst->posCurr.x;
		pInst->boundingBox.max.y = 1.0f / 2.0f * pInst->scale.y + pInst->posCurr.y;

		pInst->posCurr.x = pInst->velCurr.x * g_dt + pInst->posPrev.x;
		pInst->posCurr.y = pInst->velCurr.y * g_dt + pInst->posPrev.y;
	}


	//Check for grid collision
	for(i = 0; i < GAME_OBJ_INST_NUM_MAX; ++i)
	{
		pInst = sGameObjInstList + i;

		// skip non-active object instances and particle instances
		if ((pInst->flag & FLAG_ACTIVE) == 0 || pInst->pObject->type == TYPE_OBJECT_PARTICLE)
			continue;

		/*************
		Update grid collision flag

		if collision from bottom
			Snap to cell on Y axis
			Velocity Y = 0

		if collision from top
			Snap to cell on Y axis
			Velocity Y = 0
	
		if collision from left
			Snap to cell on X axis
			Velocity X = 0

		if collision from right
			Snap to cell on X axis
			Velocity X = 0
		*************/

		pInst->gridCollisionFlag = CheckInstanceBinaryMapCollision(pInst->posCurr.x, pInst->posCurr.y, pInst->scale.x, pInst->scale.y);

		if (pInst->gridCollisionFlag & COLLISION_BOTTOM || pInst->gridCollisionFlag & COLLISION_TOP)
		{
			SnapToCell(&pInst->posCurr.y);
			pInst->velCurr.y = 0.f;
		}

		if (pInst->gridCollisionFlag & COLLISION_LEFT || pInst->gridCollisionFlag & COLLISION_RIGHT)
		{
			SnapToCell(&pInst->posCurr.x);
			pInst->velCurr.x = 0.f;
		}
	}


	//Checking for collision among object instances:
	//Hero against enemies
	//Hero against coins

	/**********
	for each game object instance
		Skip if it's inactive or if it's non collidable

		If it's an enemy
			If collision between the enemy instance and the hero (rectangle - rectangle)
				Decrement hero lives
				Reset the hero's position in case it has lives left, otherwise RESTART the level

		If it's a coin
			If collision between the coin instance and the hero (rectangle - rectangle)
				Remove the coin and decrement the coin counter.
				Go to level2, in case no more coins are left and you are at Level1.
				Quit the game level to the main menu, in case no more coins are left and you are at Level2.
	**********/
	
	for(i = 0; i < GAME_OBJ_INST_NUM_MAX; ++i)
	{
		pInst = sGameObjInstList + i;


		// skip non-active object instances
		if ((pInst->flag & FLAG_ACTIVE) == 0)
			continue;

		if (pInst->pObject->type == TYPE_OBJECT_ENEMY1)
		{
			if (CollisionIntersection_RectRect(pInst->boundingBox, pInst->velCurr, pHero->boundingBox, pHero->velCurr))
			{
				--HeroLives;
				if (HeroLives > 0)
				{
					pHero->posCurr.x = static_cast<f32>(Hero_Initial_X) + pHero->scale.x / 2;
					pHero->posCurr.y = static_cast<f32>(Hero_Initial_Y) + pHero->scale.y / 2;
				}
				else
				{
					gGameStateNext = GS_RESTART;
				}
			}
		}

		if (pInst->pObject->type == TYPE_OBJECT_COIN)
		{
			pInst->boundingBox.min.x += 0.01f;
			pInst->boundingBox.min.y += 0.01f;	
			pInst->boundingBox.max.y -= 0.01f;	
			pInst->boundingBox.max.y -= 0.01f;

			if (CollisionIntersection_RectRect(pInst->boundingBox, pInst->velCurr, pHero->boundingBox, pHero->velCurr))
			{
				gameObjInstDestroy(pInst);
				--TotalCoins;
				if (TotalCoins == 0 && level == 1)
				{
					level = 2;
					gGameStateNext = GS_RESTART;
				}
				else if (TotalCoins == 0 && level == 2)
				{
					gGameStateNext = GS_MENU;
				}
			}
		}
	}

	
	//Computing the transformation matrices of the game object instances
	for(i = 0; i < GAME_OBJ_INST_NUM_MAX; ++i)
	{
		AEMtx33 scale, rot, trans;
		pInst = sGameObjInstList + i;

		// skip non-active object
		if ((pInst->flag & FLAG_ACTIVE) == 0)
			continue;

		AEMtx33Scale(&scale, pInst->scale.x, pInst->scale.y);
		AEMtx33Rot(&rot, pInst->dirCurr);
		AEMtx33Trans(&trans, pInst->posCurr.x - pInst->scale.x / 2, pInst->posCurr.y - pInst->scale.y / 2);

		AEMtx33Concat(&pInst->transform, &rot, &scale);
		AEMtx33Concat(&pInst->transform, &trans, &pInst->transform);
	}


	// Update Camera position, for Level2
		// To follow the player's position
		// To clamp the position at the level's borders, between (0,0) and and maximum camera position that you need to calculate
			// You may use an alpha engine helper function to clamp the camera position: AEClamp()
			// to set camera position use AEGfxSetCamPosition()

	if (level == 2)
	{
		AEGfxSetCamPosition(AEClamp(pHero->posCurr.x * MapScale.x + MapTranslate.x, 0.f, static_cast<f32>(AEGfxGetWindowWidth())),
			AEClamp(pHero->posCurr.y * MapScale.y + MapTranslate.y, 0.f, static_cast<f32>(AEGfxGetWindowHeight())));
	}
}

/******************************************************************************/
/*!
	"Draw" function of this state
*/
/******************************************************************************/
void GameStatePlatformDraw(void)
{
	//Drawing the tile map (the grid)
	int i, j;
	AEMtx33 cellTranslation{ 0 }, cellFinalTransformation{ 0 };

	UNREFERENCED_PARAMETER(cellTranslation);
	UNREFERENCED_PARAMETER(cellFinalTransformation);

	AEGfxSetBackgroundColor(0.f, 0.f, 0.f);
	AEGfxSetRenderMode(AE_GFX_RM_COLOR);
	AEGfxSetColorToMultiply(1.0f, 1.0f, 1.0f, 1.0f);
	AEGfxSetBlendMode(AE_GFX_BM_BLEND);
	AEGfxSetTransparency(1.0f);

	//Drawing the tile map

	/******REMINDER*****
	You need to concatenate MapTransform with the transformation matrix 
	of any object you want to draw. MapTransform transform the instance 
	from the normalized coordinates system of the binary map
	*******************/

	/*********
	for each array element in BinaryCollisionArray (2 loops)
		Compute the cell's translation matrix acoording to its 
		X and Y coordinates and save it in "cellTranslation"
		Concatenate MapTransform with the cell's transformation 
		and save the result in "cellFinalTransformation"
		Send the resultant matrix to the graphics manager using "AEGfxSetTransform"

		Draw the instance's shape depending on the cell's value using "AEGfxMeshDraw"
			Use the black instance in case the cell's value is TYPE_OBJECT_EMPTY
			Use the white instance in case the cell's value is TYPE_OBJECT_COLLISION
	*********/
	for (i = 0; i < BINARY_MAP_WIDTH; ++i)
	{
		for (j = 0; j < BINARY_MAP_HEIGHT; ++j)
		{
			AEMtx33Trans(&cellTranslation, static_cast<f32>(i), static_cast<f32>(j));
			AEMtx33Concat(&cellFinalTransformation, &MapTransform, &cellTranslation);
			AEGfxSetTransform(cellFinalTransformation.m);

			switch (GetCellValue(i, j))
			{
				case TYPE_OBJECT_EMPTY:
					AEGfxMeshDraw(pBlackInstance->pObject->pMesh, AE_GFX_MDM_TRIANGLES);
					break;
				case TYPE_OBJECT_COLLISION:
					AEGfxMeshDraw(pWhiteInstance->pObject->pMesh, AE_GFX_MDM_TRIANGLES);
					break;
			}

		}
	}
		



	//Drawing the object instances
	/**********
	For each active and visible object instance
		Concatenate MapTransform with its transformation matrix
		Send the resultant matrix to the graphics manager using "AEGfxSetTransform"
		Draw the instance's shape using "AEGfxMeshDraw"
	**********/
	for (i = 0; i < GAME_OBJ_INST_NUM_MAX; i++)
	{
		GameObjInst* pInst = sGameObjInstList + i;

		// skip non-active object
		if (0 == (pInst->flag & FLAG_ACTIVE) || 0 == (pInst->flag & FLAG_VISIBLE))
			continue;
		
		//Don't forget to concatenate the MapTransform matrix with the transformation of each game object instance
		AEMtx33Concat(&pInst->transform, &MapTransform, &pInst->transform);
		AEGfxSetTransform(pInst->transform.m);
		AEGfxMeshDraw(pInst->pObject->pMesh, AE_GFX_MDM_TRIANGLES);
	}

	char strBuffer[100];
	memset(strBuffer, 0, 100*sizeof(char));
	sprintf_s(strBuffer, "Lives:  %i", HeroLives);
	//printf("Player Pos: %f, %f\n", pHero->posCurr.x, pHero->posCurr.y);
	//AEGfxPrint(650, 30, (u32)-1, strBuffer);

	// Display number of coins left
	std::string total_coins = "Coins Left:  " + std::to_string(TotalCoins);
	AEGfxPrint(g_font, total_coins.c_str(), -0.9f, 0.9f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f);

	// Display number of lives left
	std::string total_lives = "Lives:  " + std::to_string(HeroLives);
	AEGfxPrint(g_font, total_lives.c_str(), 0.6f, 0.9f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f);

}

/******************************************************************************/
/*!
	"Free" function of this state
*/
/******************************************************************************/
void GameStatePlatformFree(void)
{
	// kill all object in the list
	for (unsigned int i = 0; i < GAME_OBJ_INST_NUM_MAX; i++)
		gameObjInstDestroy(sGameObjInstList + i);

	/*********
	Free the map data
	*********/
	FreeMapData();
}

/******************************************************************************/
/*!
	"Unload" function of this state
*/
/******************************************************************************/
void GameStatePlatformUnload(void)
{
	// free all CREATED mesh
	for (u32 i = 0; i < sGameObjNum; i++)
		AEGfxMeshFree(sGameObjList[i].pMesh);

	// free object list and object instances list
	free(sGameObjList);
	free(sGameObjInstList);
}

/******************************************************************************/
/*!
	Function that creates a game instance and add it to sGameObjInstList
*/
/******************************************************************************/
GameObjInst* gameObjInstCreate(unsigned int type, 
								AEVec2* scale,
								AEVec2* pPos, 
								AEVec2* pVel, 
								float dir, 
								enum STATE startState)
{
	AEVec2 zero;
	AEVec2Zero(&zero);

	AE_ASSERT_PARM(type < sGameObjNum);
	
	// loop through the object instance list to find a non-used object instance
	for (unsigned int i = 0; i < GAME_OBJ_INST_NUM_MAX; i++)
	{
		GameObjInst* pInst = sGameObjInstList + i;

		// check if current instance is not used
		if (pInst->flag == 0)
		{
			// it is not used => use it to create the new instance
			pInst->pObject			 = sGameObjList + type;
			pInst->flag				 = FLAG_ACTIVE | FLAG_VISIBLE;
			pInst->scale			 = *scale;
			pInst->posCurr			 = pPos ? *pPos : zero;
			pInst->velCurr			 = pVel ? *pVel : zero;
			pInst->dirCurr			 = dir;
			pInst->pUserData		 = 0;
			pInst->gridCollisionFlag = 0;
			pInst->state			 = startState;
			pInst->innerState		 = INNER_STATE_ON_ENTER;
			pInst->counter			 = 0;
			
			// return the newly created instance
			return pInst;
		}
	}

	return 0;
}

/******************************************************************************/
/*!
	Function that destroys a game instance
*/
/******************************************************************************/
void gameObjInstDestroy(GameObjInst* pInst)
{
	// if instance is destroyed before, just return
	if (pInst->flag == 0)
		return;

	// zero out the flag
	pInst->flag = 0;
}

/******************************************************************************/
/*!
	Function that returns the value of a cell in BinaryCollisionArray
*/
/******************************************************************************/
int GetCellValue(int X, int Y)
{
	if (X < 0 || Y < 0 || X >= BINARY_MAP_WIDTH || Y >= BINARY_MAP_HEIGHT)
	{
		return 0;
	}

	return BinaryCollisionArray[X][Y];
}

/******************************************************************************/
/*!
	Function that checks for instance binary map collision
*/
/******************************************************************************/
int CheckInstanceBinaryMapCollision(float PosX, float PosY, float scaleX, float scaleY)
{
	// collision flag;
	int FLAG{ 0b0000 };

	// x,y coordinates for hotspots
	float x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8;

	// top side
	// left top side
	x1 = PosX - scaleX / 4;
	y1 = PosY + scaleY / 2;

	// right top side
	x2 = PosX + scaleX / 4;
	y2 = PosY + scaleY / 2;

	if (GetCellValue(static_cast<int>(x1), static_cast<int>(y1)) ||
		GetCellValue(static_cast<int>(x2), static_cast<int>(y2)))
	{
		FLAG |= COLLISION_TOP;
	}

	// bottom side
	// left bottom side
	x3 = PosX - scaleX / 4;
	y3 = PosY - scaleY / 2;

	// right bottom side
	x4 = PosX + scaleX / 4;
	y4 = PosY - scaleY / 2;

	if (GetCellValue(static_cast<int>(x3), static_cast<int>(y3)) ||
		GetCellValue(static_cast<int>(x4), static_cast<int>(y4)))
	{
		FLAG |= COLLISION_BOTTOM;
	}

	// left side
	// upper left side
	x5 = PosX - scaleX / 2;
	y5 = PosY + scaleY / 4;

	// lower left side
	x6 = PosX - scaleX / 2;
	y6 = PosY - scaleY / 4;

	if (GetCellValue(static_cast<int>(x5), static_cast<int>(y5)) ||
		GetCellValue(static_cast<int>(x6), static_cast<int>(y6)))
	{
		FLAG |= COLLISION_LEFT;
	}

	// right side
	// upper right side
	x7 = PosX + scaleX / 2;
	y7 = PosY + scaleY / 4;

	// lower right side
	x8 = PosX + scaleX / 2;
	y8 = PosY - scaleY / 4;

	if (GetCellValue(static_cast<int>(x7), static_cast<int>(y7)) ||
		GetCellValue(static_cast<int>(x8), static_cast<int>(y8)))
	{
		FLAG |= COLLISION_RIGHT;
	}

	return FLAG;
}

/******************************************************************************/
/*!
	Function that snaps a coordinate to cell
*/
/******************************************************************************/
void SnapToCell(float *Coordinate)
{
	*Coordinate = static_cast<int>(*Coordinate) + 0.5f;
}

/******************************************************************************/
/*!
	Function that imports map data from file into MapData and BinaryCollisionArray
*/
/******************************************************************************/
int ImportMapDataFromFile(char *FileName)
{
	std::ifstream ifs(FileName);

	if (!ifs.is_open())
	{
		return 0;
	}

	std::string width, height;

	// read width and height
	ifs >> width >> BINARY_MAP_WIDTH >> height >> BINARY_MAP_HEIGHT;

	// dynamically allocate memory for 2D MapData and BinaryCollisionArray
	MapData = new int* [BINARY_MAP_WIDTH];
	BinaryCollisionArray = new int* [BINARY_MAP_WIDTH];

	for (int i{}; i < BINARY_MAP_WIDTH; ++i)
	{
		MapData[i] = new int[BINARY_MAP_HEIGHT];
		BinaryCollisionArray[i] = new int[BINARY_MAP_HEIGHT];
	}

	// read data from file into MapData and BinaryCollisionArray
	for (int i{}; i < BINARY_MAP_HEIGHT; ++i)
	{
		for (int j{}; j < BINARY_MAP_WIDTH; ++j)
		{
			ifs >> MapData[j][i];
			BinaryCollisionArray[j][i] = MapData[j][i];

			if (BinaryCollisionArray[j][i] != 1 && BinaryCollisionArray[j][i] != 0)
			{
				BinaryCollisionArray[j][i] = 0;
			}
		}
	}

	ifs.close();
	return 1;
}

/******************************************************************************/
/*!
	Function that frees map data
*/
/******************************************************************************/
void FreeMapData()
{
	for (int i{}; i < BINARY_MAP_WIDTH; ++i)
	{
		delete[] MapData[i];
		delete[] BinaryCollisionArray[i];
	}

	delete[] MapData;
	delete[] BinaryCollisionArray;
}

/******************************************************************************/
/*!
	State machine for the enemies
*/
/******************************************************************************/
void EnemyStateMachine(GameObjInst *pInst)
{
	/***********
	This state machine has 2 states: STATE_GOING_LEFT and STATE_GOING_RIGHT
	Each state has 3 inner states: INNER_STATE_ON_ENTER, INNER_STATE_ON_UPDATE, INNER_STATE_ON_EXIT
	Use "switch" statements to determine which state and inner state the enemy is currently in.

	
	STATE_GOING_LEFT
		INNER_STATE_ON_ENTER
			Set velocity X to -MOVE_VELOCITY_ENEMY
			Set inner state to "on update"

		INNER_STATE_ON_UPDATE
			If collision on left side OR bottom left cell is non collidable
				Initialize the counter to ENEMY_IDLE_TIME
				Set inner state to on exit
				Set velocity X to 0


		INNER_STATE_ON_EXIT
			Decrement counter by frame time
			if counter is less than 0 (sprite's idle time is over)
				Set state to "going right"
				Set inner state to "on enter"

	STATE_GOING_RIGHT is basically the same, with few modifications.

	***********/
	switch (pInst->state)
	{
		case STATE_GOING_LEFT:
			switch (pInst->innerState)
			{
				case INNER_STATE_ON_ENTER:
					pInst->velCurr.x = -MOVE_VELOCITY_ENEMY;
					pInst->innerState = INNER_STATE_ON_UPDATE;
					break;
				case INNER_STATE_ON_UPDATE:
					if (pInst->gridCollisionFlag & COLLISION_LEFT || 
						GetCellValue(static_cast<int>(pInst->posCurr.x - pInst->scale.x / 2), static_cast<int>(pInst->posCurr.y - pInst->scale.y)) == TYPE_OBJECT_EMPTY)
					{
						pInst->velCurr.x = 0.f;
						pInst->counter = ENEMY_IDLE_TIME;
						pInst->innerState = INNER_STATE_ON_EXIT;
					}
					break;
				case INNER_STATE_ON_EXIT:
					pInst->counter -= g_dt;
					if (pInst->counter <= 0.0)
					{
						pInst->state = STATE_GOING_RIGHT;
						pInst->innerState = INNER_STATE_ON_ENTER;
					}
					break;
			}
			break;

		case STATE_GOING_RIGHT:
			switch (pInst->innerState)
			{
				case INNER_STATE_ON_ENTER:
					pInst->velCurr.x = MOVE_VELOCITY_ENEMY;
					pInst->innerState = INNER_STATE_ON_UPDATE;
					break;
				case INNER_STATE_ON_UPDATE:
					if (pInst->gridCollisionFlag & COLLISION_RIGHT || 
						GetCellValue(static_cast<int>(pInst->posCurr.x + pInst->scale.x / 2), static_cast<int>(pInst->posCurr.y - pInst->scale.y)) == TYPE_OBJECT_EMPTY)
					{
						pInst->velCurr.x = 0.f;
						pInst->counter = ENEMY_IDLE_TIME;
						pInst->innerState = INNER_STATE_ON_EXIT;	
					}
					break;
				case INNER_STATE_ON_EXIT:
					pInst->counter -= g_dt;
					if (pInst->counter <= 0.0)
					{
						pInst->state = STATE_GOING_LEFT;
						pInst->innerState = INNER_STATE_ON_ENTER;
					}
					break;
			}
			break;
	}
}