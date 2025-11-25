/******************************************************************************/
/*!
\file		GameState_Cage.cpp
\author 	Bryan Ang Wei Ze, bryanweize.ang, 2301397
\par    	bryanweize.ang@digipen.edu
\date   	Mar 22 2024
\brief  	This file implements the Load, Init, Update, Draw, Free and Unload
			functions of GameState_Cage

Copyright (C) 2024 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the prior
written consent of DigiPen Institute of Technology is prohibited.
*/
/******************************************************************************/

#include "main.h"

/******************************************************************************/
/*!
	Defines
*/
/******************************************************************************/
const unsigned int	GAME_OBJ_NUM_MAX		= 32;	//The total number of different objects (Shapes)
const unsigned int	GAME_OBJ_INST_NUM_MAX	= 2048;	//The total number of different game object instances

//Flags
const unsigned int	FLAG_ACTIVE				= 0x00000001;
const unsigned int	FLAG_VISIBLE			= 0x00000002;
const unsigned int	FLAG_NON_COLLIDABLE		= 0x00000004;

const float			PI_OVER_180				= PI/180.0f;


//values: 0,1
//0: original: no extra credits
//1: Extra Credits: with line edges collision

int EXTRA_CREDITS = 1;



enum class TYPE_OBJECT
{
	TYPE_OBJECT_BALL,	//0
	TYPE_OBJECT_WALL,	//1
	TYPE_OBJECT_PILLAR, //2

	TYPE_OBJECT_NUM
};

/******************************************************************************/
/*!
	Struct/Class Definitions
*/
/******************************************************************************/
struct GameObj
{
	TYPE_OBJECT	type;		// object type
	AEGfxVertexList *	pMesh;		// pbject
};


struct GameObjInst
{
	GameObj*		pObject;	// pointer to the 'original'
	unsigned int	flag;		// bit flag or-ed together
	float			scale;
	CSD1130::Vec2	posCurr;	// object current position
	CSD1130::Vec2	velCurr;	// object current velocity
	float			dirCurr;	// object current direction
	float			speed;

	CSD1130::Mtx33	transform;	// object drawing matrix

	// pointer to custom data specific for each object type
	void*			pUserData;
};


/******************************************************************************/
/*!
	File globals
*/
/******************************************************************************/
// list of original objects
static GameObj			*sGameObjList;
static unsigned int		sGameObjNum;

// list of object instances
static GameObjInst		*sGameObjInstList;
static unsigned int		sGameObjInstNum;

// function to create/destroy a game object instance
GameObjInst*		gameObjInstCreate (TYPE_OBJECT type,
										   float scale, 
										   CSD1130::Vec2* pPos, 
										   CSD1130::Vec2* pVel, 
										   float dir);
void				gameObjInstDestroy(GameObjInst* pInst);

static Circle *sBallData = 0;
static LineSegment *sWallData = 0;



/******************************************************************************/
/*!
	
*/
/******************************************************************************/
void GameStateCageLoad(void)
{
	//validating
	if (EXTRA_CREDITS > 1 || EXTRA_CREDITS < 0)
		EXTRA_CREDITS = 0;

	sGameObjList = (GameObj *)calloc(GAME_OBJ_NUM_MAX, sizeof(GameObj));
	sGameObjInstList = (GameObjInst *)calloc(GAME_OBJ_INST_NUM_MAX, sizeof(GameObjInst));
	sGameObjNum = 0;

	GameObj* pObj;

	//------------------------------------------

	//Creating the ball object
	pObj		= sGameObjList + sGameObjNum++;
	pObj->type	= TYPE_OBJECT::TYPE_OBJECT_BALL;

	AEGfxMeshStart();

	//1st argument: X
	//2nd argument: Y
	//3rd argument: ARGB

	//Creating the ball shape
	int Parts = 36;
	for(float i = 0; i < Parts; ++i)
	{
		AEGfxTriAdd(
		0.0f, 0.0f, 0xFFFFFF00, 0.0f, 0.0f,
		cosf(i*2*PI/Parts)*1.0f,  sinf(i*2*PI/Parts)*1.0f, 0xFFFFFF00, 0.0f, 0.0f,
		cosf((i+1)*2*PI/Parts)*1.0f,  sinf((i+1)*2*PI/Parts)*1.0f, 0xFFFFFF00, 0.0f, 0.0f);
	}

	pObj->pMesh = AEGfxMeshEnd();

	//------------------------------------------

	// Creating the wall object
	pObj		= sGameObjList + sGameObjNum++;
	pObj->type	= TYPE_OBJECT::TYPE_OBJECT_WALL;

	AEGfxMeshStart();

	//1st argument: X
	//2nd argument: Y
	//3rd argument: ARGB

	//Creating the wall shape
	AEGfxVertexAdd(-0.5f, 0.0f, 0xFFFF0000, 0.0f, 0.0f);
	AEGfxVertexAdd(0.5f, 0.0f, 0xFFFFFFFF, 0.0f, 0.0f);

	pObj->pMesh = AEGfxMeshEnd();

	//------------------------------------------

	

	AEGfxSetBackgroundColor(0.2f, 0.2f, 0.2f);

	
}

/******************************************************************************/
/*!

*/
/******************************************************************************/
void GameStateCageInit(void)
{
	GameObjInst *pInst;
	std::string str;
	std::ifstream inFile;

	if(EXTRA_CREDITS == 0)
		inFile.open("..\\Bin\\Resources\\LevelData - Original.txt");
	else if (EXTRA_CREDITS == 1)
		inFile.open("..\\Bin\\Resources\\LevelData - Extra Credits.txt");
	
	
	if(inFile.is_open())
	{
		// read ball data
		float dir, speed, scale;
		unsigned int ballNum = 0;
		inFile>>ballNum;
		sBallData = new Circle[ballNum];

		for(unsigned int i = 0; i < ballNum; ++i)
		{
			// read pos
			inFile>>str>>sBallData[i].m_center.x;
			inFile>>str>>sBallData[i].m_center.y;
			// read direction
			inFile>>str>>dir;
			// read speed
			inFile>>str>>speed;
			// read radius
			inFile>>str>>sBallData[i].m_radius;
			
			// create ball instance
			CSD1130::Vec2 vel;
			//AEVec2Set(&vel, cos(dir*PI_OVER_180)*speed, 
							//sin(dir*PI_OVER_180)*speed);
			vel.x = cos(dir * PI_OVER_180) * speed;
			vel.y = sin(dir * PI_OVER_180) * speed;
			pInst = gameObjInstCreate(TYPE_OBJECT::TYPE_OBJECT_BALL, sBallData[i].m_radius,
										&sBallData[i].m_center, &vel, 0.0f);
			AE_ASSERT(pInst);
			pInst->speed = speed;
			pInst->pUserData = &sBallData[i];
		}

		// read wall data
		unsigned int wallNum = 0;
		CSD1130::Vec2 P0 = CSD1130::Vec2(), P1 = CSD1130::Vec2();
		CSD1130::Vec2 pos = CSD1130::Vec2(), e = CSD1130::Vec2();

		inFile>>wallNum;
		sWallData = new LineSegment[wallNum];

		for(unsigned int i = 0; i < wallNum; ++i)
		{
			inFile>>str>> P0.x;
			inFile>>str>> P0.y;
			inFile>>str>> P1.x;
			inFile>>str>> P1.y;

			//stupid work
			pos.x = (P0.x + P1.x) * 0.5f;
			pos.y = (P0.y + P1.y) * 0.5f;
			e.x = P1.x - P0.x;
			e.y = P1.y - P0.y;
			scale = sqrtf((e.x * e.x) + (e.y * e.y));
			//a.b = |a|*|b|*cos(a,b)
			float cosine = (e.x/* * 1.0f + e.y * 0.0f*/) / (scale);//assuming scale is non-zero (controlling our data input!)
			float acosine = acos(cosine);
			if (e.y < 0.0f)
				acosine = 2*PI - acosine;



			//NOTE to student: When using your own build line segment function, comment out this next line, and uncomment the line after
			//AEBuildLineSegment(sWallData[i], pos, scale, acosine);
			BuildLineSegment(sWallData[i], P0, P1);



			pInst = gameObjInstCreate(TYPE_OBJECT::TYPE_OBJECT_WALL, scale, &pos, 0, acosine);
			AE_ASSERT(pInst);
			pInst->pUserData = &sWallData[i];
		}

		

		inFile.clear();
		inFile.close();
	}
	else
	{
		//AE_ASSERT_MESG(inFile, "Failed to open the text file");
		printf("Failed to open the text file");
	}
}



/******************************************************************************/
/*!

*/
/******************************************************************************/
void GameStateCageUpdate(void)
{
	static bool full_screen_me;
	if (AEInputCheckTriggered(AEVK_F))
	{
		full_screen_me = !full_screen_me;
		AESysSetFullScreen((s32)full_screen_me);
	}

	
	CSD1130::Vec2		interPtA;
	CSD1130::Vec2      normalAtCollision;
	float		interTime = 0.0f;

	//f32 fpsT = (f32)AEFrameRateControllerGetFrameTime();

	//Update object instances positions
	for(unsigned int i = 0; i < GAME_OBJ_INST_NUM_MAX; ++i)
	{
		GameObjInst *pBallInst = sGameObjInstList + i;

		// skip non-active object
		if (0 == (pBallInst->flag & FLAG_ACTIVE) ||
			pBallInst->pObject->type != TYPE_OBJECT::TYPE_OBJECT_BALL)
			continue;

		CSD1130::Vec2 posNext;
		posNext.x = pBallInst->posCurr.x + pBallInst->velCurr.x * g_dt;
		posNext.y = pBallInst->posCurr.y + pBallInst->velCurr.y * g_dt;

		// Update the latest ball data with the lastest ball's position
		Circle &ballData = *((Circle*)pBallInst->pUserData);
		ballData.m_center.x = pBallInst->posCurr.x;
		ballData.m_center.y = pBallInst->posCurr.y;

		// Check collision with walls
		for(unsigned int j = 0; j < GAME_OBJ_INST_NUM_MAX; ++j)
		{
			GameObjInst *pInst = sGameObjInstList + j;

			if (0 == (pInst->flag & FLAG_ACTIVE))
				continue;

			switch(pInst->pObject->type)
			{
				case TYPE_OBJECT::TYPE_OBJECT_WALL:
				{
					LineSegment &lineSegData = *((LineSegment*)pInst->pUserData);

					bool checkLineEdges = false;
					if (EXTRA_CREDITS == 1)
						checkLineEdges = true;
					
					if ((pBallInst->velCurr.x * lineSegData.m_normal.x + pBallInst->velCurr.y * lineSegData.m_normal.y) <= 0.0f)
					{
						if (CollisionIntersection_CircleLineSegment(ballData,
							posNext,
							lineSegData,
							interPtA,
							normalAtCollision,
							interTime,
							checkLineEdges))
						{
							CSD1130::Vec2 reflectedVec;
							CollisionResponse_CircleLineSegment(interPtA,
								normalAtCollision,
								posNext,
								reflectedVec);
							pBallInst->velCurr.x = reflectedVec.x * pBallInst->speed;
							pBallInst->velCurr.y = reflectedVec.y * pBallInst->speed;
						}
					}
				}
				break;
				
				

			}
		}

		pBallInst->posCurr.x = posNext.x;
		pBallInst->posCurr.y = posNext.y;
	}

	
	//Computing the transformation matrices of the game object instances
	for(unsigned int i = 0; i < GAME_OBJ_INST_NUM_MAX; ++i)
	{
		CSD1130::Mtx33 scale, rot, trans;
		GameObjInst *pInst = sGameObjInstList + i;

		// skip non-active object
		if (0 == (pInst->flag & FLAG_ACTIVE))
			continue;

		CSD1130::Mtx33Scale(scale, pInst->scale, pInst->scale);
		CSD1130::Mtx33RotRad(rot, pInst->dirCurr);
		CSD1130::Mtx33Translate(trans, pInst->posCurr.x, pInst->posCurr.y);

		//AEMtx33Concat(&pInst->transform, &scale, &rot);
		//AEMtx33Concat(&pInst->transform, &trans, &pInst->transform);
		pInst->transform = scale * rot;
		pInst->transform = trans * pInst->transform;
	}

	if(AEInputCheckTriggered(AEVK_R))
		gGameStateNext = GS_STATE::GS_RESTART;
}

/******************************************************************************/
/*!

*/
/******************************************************************************/
void GameStateCageDraw(void)
{
	AEGfxSetBlendMode(AE_GFX_BM_BLEND);
	
	AEGfxSetRenderMode(AE_GFX_RM_COLOR);
	AEGfxTextureSet(NULL, 0, 0);
	AEGfxSetTransparency(1.0f);

	
	//Drawing the object instances
	int only4 = 0;
	for (unsigned int i = 0; i < GAME_OBJ_INST_NUM_MAX; i++)
	{
		GameObjInst* pInst = sGameObjInstList + i;

		// skip non-active object
		if (0 == (pInst->flag & FLAG_ACTIVE) || 0 == (pInst->flag & FLAG_VISIBLE))
			continue;
		
		AEGfxSetTransform(pInst->transform.m2);

		if (pInst->pObject->type == TYPE_OBJECT::TYPE_OBJECT_BALL)
		{
			int ttiimmee = (int)timeGetTime();
			ttiimmee %= 5;

			++only4;
			if (only4 > 4)
			{
				AEGfxSetColorToMultiply(1.0f * cosf((float)(i * 2) * PI / (float)ttiimmee), 1.0f, 0.0f, 1.0f);
			}
			else
			{
				AEGfxSetColorToMultiply(1.0f, 0.2f, 0.2f, 1.0f);
			}
			AEGfxMeshDraw(pInst->pObject->pMesh, AE_GFX_MDM_TRIANGLES);
		}
		else if (pInst->pObject->type == TYPE_OBJECT::TYPE_OBJECT_WALL)
		{
			AEGfxSetColorToMultiply(1.0f, 1.0f, 1.0f, 1.0f);
			AEGfxMeshDraw(pInst->pObject->pMesh, AE_GFX_MDM_LINES_STRIP);
		}
	}
	
	char strBuffer[100];
	memset(strBuffer, 0, 100*sizeof(char));
	sprintf_s(strBuffer, "FPS:  %.6f", 1.0f / AEFrameRateControllerGetFrameTime());
	
	
	AEGfxSetRenderMode(AE_GFX_RM_COLOR);
	AEGfxSetBlendMode(AE_GFX_BM_BLEND);
	AEGfxTextureSet(NULL, 0, 0);	
	AEGfxSetTransparency(1.0f);
	
	//AEGfxPrint(fontId, strBuffer, -0.95f, -0.95f, 2.0f, 1.f, 0.f, 1.f);
	AEGfxPrint(fontId, strBuffer, (270.0f) / (float)(AEGfxGetWindowWidth() / 2), (350.0f) / (float)(AEGfxGetWindowHeight() / 2), 1.0f, 1.0f, 0.0f, 0.0f, 1.0f);
}

/******************************************************************************/
/*!

*/
/******************************************************************************/
void GameStateCageFree(void)
{
	// kill all object in the list
	for (unsigned int i = 0; i < GAME_OBJ_INST_NUM_MAX; i++)
		gameObjInstDestroy(sGameObjInstList + i);

	delete []sBallData;
	sBallData = NULL;
	
	delete []sWallData;
	sWallData = NULL;

}

/******************************************************************************/
/*!

*/
/******************************************************************************/
void GameStateCageUnload(void)
{
	// free all CREATED mesh
	for (u32 i = 0; i < sGameObjNum; i++)
		AEGfxMeshFree(sGameObjList[i].pMesh);

	free(sGameObjInstList);
	free(sGameObjList);
}

/******************************************************************************/
/*!

*/
/******************************************************************************/
GameObjInst* gameObjInstCreate(TYPE_OBJECT type,
							   float scale, 
							   CSD1130::Vec2* pPos, 
							   CSD1130::Vec2* pVel, 
							   float dir)
{
	CSD1130::Vec2 zero{ 0.f,0.f };
	//AEVec2Zero(&zero);

	AE_ASSERT_PARM(type < TYPE_OBJECT(sGameObjNum));
	
	// loop through the object instance list to find a non-used object instance
	for (unsigned int i = 0; i < GAME_OBJ_INST_NUM_MAX; i++)
	{
		GameObjInst* pInst = sGameObjInstList + i;

		// check if current instance is not used
		if ((pInst->flag & FLAG_ACTIVE) == 0)
		{
			// it is not used => use it to create the new instance
			pInst->pObject			 = sGameObjList + (int)type;
			pInst->flag				 = FLAG_ACTIVE | FLAG_VISIBLE;
			pInst->scale			 = scale;
			pInst->posCurr			 = pPos ? *pPos : zero;
			pInst->velCurr			 = pVel ? *pVel : zero;
			pInst->dirCurr			 = dir;
			pInst->pUserData		 = 0;
			
			// return the newly created instance
			return pInst;
		}
	}

	return 0;
}

/******************************************************************************/
/*!

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