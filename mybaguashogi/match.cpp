#include "match.h"
MatchGame::MatchGame(NetWork* black, NetWork* white, int simulation, bool gpu) :
	current(9, 9), ended(false), black(black), white(white), simulation(simulation) {}
MatchGame::MatchGame(const MatchGame& b) = default;
int MatchGame::playUntilEnd()
{
	//cout << &current.repetition[0] << ' ' << &current.repetition[1] << endl;
	//current.repetition[0][1] = 1;

	MCTS mcts[]{ MCTS(black, 3, simulation, current.get_action_size(), 0.08f, false, true, false),
		MCTS(white, 3, simulation, current.get_action_size(), 0.08f, false, true, false) };
	if (!ended)
	{
		std::vector<Player> pla_hist;
		int pla = 0;
		while (current.winner == C_WALL)
		{
			std::vector<unsigned> visits, visits_pruned;
			if (current.genLegalPolicy().empty()) { current.winner = getOpp(current.nextPla); break; }
			if (current.movenum >= 500 && !current.attacked)
			{
				current.winner = C_EMPTY;
				break;
			}
			mcts[pla].get_action_visits(current, visits, visits_pruned);
			unsigned sum = std::reduce(std::execution::par_unseq, visits_pruned.cbegin(), visits_pruned.cend());
			if (!sum) { current.winner = getOpp(current.nextPla); break; }

			std::vector<float> probs(visits_pruned.size(), 0);
			for (int i = 0; i < visits_pruned.size(); ++i)
				probs[i] = pow((float)visits_pruned[i] / sum, 1 / .8);

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