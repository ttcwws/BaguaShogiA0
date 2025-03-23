#include "match.h"
MatchGame::MatchGame(NetWork* model1, NetWork* model2, bool gpu) :
	current(9, 9), ended(false), model1(model1), model2(model2) {}
MatchGame::MatchGame(const MatchGame& b) = default;
int MatchGame::playUntilEnd()
{
	//cout << &current.repetition[0] << ' ' << &current.repetition[1] << endl;
	//current.repetition[0][1] = 1;

	MCTS mcts[]{ MCTS(model1, 17, 3, 900, .9, current.get_action_size(), 0.08, true, use_gpu, false),
		MCTS(model2, 17, 3, 900, .9, current.get_action_size(), 0.08, true, use_gpu, false) };
	if (!ended)
	{
		std::vector<Player> pla_hist;
		int pla = 0;
		while (current.winner == C_WALL)
		{
			std::vector<unsigned> visits, visits_pruned;
			mcts[pla].get_action_visits(&current, visits, visits_pruned);
			float sum = std::accumulate(visits_pruned.cbegin(), visits_pruned.cend(), 0.);
			if (abs(sum) < FLT_EPSILON) { current.winner = getOpp(current.nextPla); break; }

			std::vector<float> probs(visits_pruned.size(), 0);
			for (int i = 0; i < visits_pruned.size(); ++i)
				probs[i] = pow(visits_pruned[i] / sum, 1.2);

			//std::vector<Move> mvs;
			//current.genLegalMove(mvs, current.nextPla);
			auto policy = chooseWithProbability(probs);
			current.playPolicy(policy);
			current.checkConsistency();
			mcts[0].update_with_move(policy);
			mcts[1].update_with_move(policy);
			pla = 1 - pla;
		}
		ended = true;
	}
	if (current.winner == C_EMPTY)return 0;
	return current.winner == P_BLACK ? 1 : 2;
}