#pragma once
#include "search/mcts.h"
#include "game/board.h"
#include "search/utils.h"
#include <cmath>

class SelfGame
{
public:
	SelfGame(NetWork* model, long long seed, double ratio, unsigned start, bool gpu = true);
	SelfGame(const SelfGame& b) = delete;
	void playUntilEnd();
	// states, policy targets, value targets
	void exportTrainingData(std::vector<Piece>& bin, std::vector<float>& overall, std::vector<float>& policy, std::vector<int8_t>& value);
	Board current;
	bool ended, use_gpu;
	int batch_num;
	double record_ratio;
	Rand rand;
	std::vector<Piece> state_color_hist;
	std::vector<int8_t> value_hist;
	std::vector<float> state_overall_hist, policy_hist;
	NetWork* model;
	unsigned start;
	bool copied;
};
