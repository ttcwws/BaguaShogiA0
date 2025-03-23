#pragma once
#include "search/mcts.h"
#include "game/board.h"


class SelfGame
{
public:
	SelfGame(NetWork* model, long long seed, double ratio, unsigned start, bool gpu = true);
	SelfGame(const SelfGame& b);
	void playUntilEnd();
	// states, policy targets, value targets
	std::tuple<std::pair<Tensor, Tensor>, Tensor, Tensor> exportTrainingData();
	Board current;
	bool ended, use_gpu;
	int batch_num;
	double record_ratio;
	Rand rand;
	std::vector<Tensor> state_bin_hist, state_overall_hist;
	std::vector<Tensor> policy_hist;
	std::vector<float> value_hist;
	NetWork* model;
	unsigned start;
};