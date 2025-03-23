#pragma once
#include "search/mcts.h"
#include "game/board.h"


class MatchGame
{
public:
	MatchGame(NetWork* model1, NetWork* model2, bool gpu = true);
	MatchGame(const MatchGame& b);
	int playUntilEnd();
	// states, policy targets, value targets
	Board current;
	bool ended, use_gpu;
	NetWork* model1, * model2;
};