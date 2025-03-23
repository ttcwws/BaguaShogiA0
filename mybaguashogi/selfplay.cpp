#include "selfplay.h"
SelfGame::SelfGame(NetWork* model, long long seed, double ratio, unsigned start, bool gpu) :
	current(9, 9), ended(false), batch_num(0), model(model), use_gpu(gpu), rand(seed), record_ratio(1), start(start) {}
SelfGame::SelfGame(const SelfGame& b) = default;
void SelfGame::playUntilEnd()
{
	//cout << &current.repetition[0] << ' ' << &current.repetition[1] << endl;
	//current.repetition[0][1] = 1;

	MCTS mcts(model, 14, 3, 211, .9, current.get_action_size(), 0.08, true, use_gpu);
	//int start = 0;
	if (ended)return;
	std::vector<Player> pla_hist;
	//cout << this << endl;
	bool rec = false;
	while (current.winner == C_WALL)
	{
		if (current.movenum < start)
		{
			//mcts.num_mcts_sims = 250;
			//auto probs = mcts.get_action_probs(&current);
			//auto policy = chooseWithProbability(probs);
			//if (chooseWithProbability(std::vector<double>{1 - record_ratio, record_ratio}, rand.nextUInt()))
			//{
			//	pla_hist.push_back(current.nextPla);
			//	policy_hist.push_back(torch::from_blob(probs.data(), { (int)probs.size() }).unsqueeze(0).clone().to(at::k
			// 
			// ));
			//	auto [x, y] = boardToState(current);
			//	state_bin_hist.push_back(x.unsqueeze(0));
			//	state_overall_hist.push_back(y.unsqueeze(0));
			//	++batch_num;
			//}
			//current.playPolicy(policy);
			//current.checkConsistency();
			//mcts.update_with_move(policy);
			//continue;
		}
		//std::vector<double> action_probs(board->get_action_size(), 0);
		//std::vector<int> probs(current.get_action_size(),0);
		//probs=current.get_legal_moves();
		//{
			//std::lock_guard<std::mutex> lock(this->lock);
		else mcts.num_mcts_sims = 433;
		if (!model)mcts.num_mcts_sims = 400;
		std::vector<unsigned> visits, visits_pruned;
		mcts.get_action_visits(&current, visits, visits_pruned);
		float sum = std::accumulate(visits.cbegin(), visits.cend(), 0.);
		if (abs(sum) < FLT_EPSILON) { current.winner = getOpp(current.nextPla); break; }

		std::vector<float> probs(visits.size(), 0);
		for (int i = 0; i < visits.size(); ++i)
			probs[i] = visits[i] / sum;

		std::vector<Move> mvs;
		current.genLegalMove(mvs, current.nextPla);
		//for (Move mv : mvs)
		//{
		//	auto pol = current.moveToPolicy(mv);
		//	//using namespace Location;
		//	//if (current.nextPla == P_WHITE)mv.rot();
		//	//cout << "played:" << movePolicy[mv]<<' '<<probs[movePolicy[mv]] << ':' << '(' << getX(mv.x, 9) + 1 << "," << getY(mv.x, 9) + 1 << ")->(" << getX(mv.y, 9) + 1 << ',' << getY(mv.y, 9) + 1 << ") " << (int)mv.z << mv.change << endl;
		//	cout << pol << ' ' << visits[pol] << ' ' << visits_pruned[pol] << endl;
		//	//current.printBoard(cout, nullptr);
		//}
		auto policy = chooseWithProbability(probs);
		//int cnt = 0, max = 0, maxi = 0;
		//for (int i = 0; i < visits.size(); ++i)
		//{
		//	if (visits_pruned[i] > FLT_EPSILON)
		//	{
		//		if (visits[i] > max)
		//			max = visits[i], maxi = i;
		//		cout << i << ":" << visits_pruned[i] << endl;
		//	}
		//}
		if (rec = chooseWithProbability(std::vector<double>{1 - record_ratio, record_ratio}, rand.nextUInt()))
		{
			std::vector<float> policy(visits_pruned.size(), 0);
			float sum_p = std::accumulate(visits_pruned.cbegin(), visits_pruned.cend(), 0.);
			//if (!std::accumulate(visits_pruned.cbegin(), visits_pruned.cend(), 0))
			//{
			//	for (int i = 0; i < visits.size(); ++i)
			//		if (visits[i])
			//			cout << i << ' ' << visits[i]<<' '<<visits_pruned[i] << endl;
			//	cout << current << endl;
			//}
			for (int i = 0; i < visits_pruned.size(); ++i)
				policy[i] = visits_pruned[i] / sum_p;
			//cout << current;
			//for (Move mv : mvs)
			//	cout << to_string(mv) << ' ' << visits_pruned[current.moveToPolicy(mv)] << ' ' << visits[current.moveToPolicy(mv)] << endl;
			//getchar();
			pla_hist.push_back(current.nextPla);
			policy_hist.push_back(torch::from_blob(policy.data(), { (int)policy.size() }).clone());
			auto [x, y] = boardToState(current, false);
			state_bin_hist.push_back(x);
			state_overall_hist.push_back(y);
			++batch_num;
		}
		current.playPolicy(policy);
		current.checkConsistency();
		mcts.update_with_move(policy);
		//cout << policy << endl;
		//cout << current.movenum << endl;
		//current.printBoard(cout, 0);
		//cout << current;
		//cout << batch_num << endl;
	}
	//if (rec && current.winner == current.nextPla)
	//{
	//	pla_hist.pop_back();
	//	policy_hist.pop_back();
	//	state_bin_hist.pop_back();
	//	state_overall_hist.pop_back();
	//	--batch_num;
	//	//cout << "YES!!!" << endl;
	//}
	//else cout << "NO!!!" << endl;
	for (const Player& pla : pla_hist)
	{
		//cout << (int)pla << ' ' << (int)current.winner << endl;
		if (current.winner == C_EMPTY)value_hist.push_back(0);
		else value_hist.push_back(current.winner == pla ? 1 : -1);
		//cout << value_hist.back() << endl;
	}
	ended = true;
}
std::tuple<std::pair<Tensor, Tensor>, Tensor, Tensor> SelfGame::exportTrainingData()
{
	return std::tuple<std::pair<Tensor, Tensor>, Tensor, Tensor>(
		std::make_pair(at::stack(state_bin_hist), at::stack(state_overall_hist)),
		at::stack(policy_hist),
		torch::from_blob(value_hist.data(), { batch_num }).clone()
	);
}