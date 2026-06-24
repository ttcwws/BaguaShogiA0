#pragma once
#include "search/mcts.h"
#include "game/board.h"
#include <execution>


class MatchGame
{
public:
	MatchGame(NetWork* black, NetWork* white, int simulation, bool gpu = true);
	MatchGame(const MatchGame& b);
	int playUntilEnd();
	// states, policy targets, value targets
	Board current;
	bool ended;
	int simulation;
	NetWork* black, * white;
};