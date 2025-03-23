#include <cstdio>
#include <iostream>
#include <random>
#include <ctime>
//#include <torch/nn.h>
#include "game/board.h"
#include "threadpool2.h"]
#include "match.h"
#include "core/rand.h"
#include "selfplay.h"
#include "cnpy.h"
using namespace torch;
using namespace torch::optim;
using std::cout;
using std::endl;
Piece parsePiece(char c)
{
	switch (c)
	{
	case 'K':
		return S_KING;
	case 'G':
		return S_GOLD;
	case 'S':
		return S_SILVER;
	case 'L':
		return S_LANCE;
	case 'B':
		return S_BISHOP;
	case 'R':
		return S_ROOK;
	case 'P':
		return S_PAWN;
	case 'N':
		return S_KNIGHT;
	default:break;
	}
	return C_EMPTY;
}
Move parseMove(std::string str)
{
	using namespace Location;
	if (str.length() > 5 || str.length() < 4)return Move(-1, -1);
	if (isalpha(str[1]))
	{
		if (!isdigit(str[2]))return Move(-1, -1, 0);
		str[0] = toupper(str[0]);
		str[1] = toupper(str[1]);
		int x1 = str[1] - 'A', y1 = str[2] - '1';
		y1 = 8 - y1;
		Piece pc = parsePiece(str[0]);
		if (str[3] != '~' && str[3] != '=')return Move(-1, -1);
		return Move(getLoc(x1, y1, 9), 0, pc, str[3] == '=');
	}
	if (!isalpha(str[0]) || !isalpha(str[2]) || !isdigit(str[1]) || !isdigit(str[3]))return Move(-1, -1, 0);
	str[0] = toupper(str[0]);
	str[2] = toupper(str[2]);
	int x1 = str[0] - 'A', x2 = str[2] - 'A', y1 = str[1] - '1', y2 = str[3] - '1';
	y1 = 8 - y1; y2 = 8 - y2;
	Loc x = getLoc(x1, y1, 9), y = getLoc(x2, y2, 9);
	bool change = false;
	if (str.length() == 5)
	{
		if (str.back() != '+')return Move(-1, -1);
		change = true;
	}
	return Move(x, y, 0, change);
}
int main(int argc, char* argv[])
{
	int num_games = 1000, num_threads = 100, num_report = 50, batch_size = 256, save_size = 1050;
	int num_search = 1000, search_threads = 1;
	////player
	//if (argc > 1)
	//	num_search = atoi(argv[1]);
	//if (argc > 2)
	//	search_threads = atoi(argv[2]);

	//match
	jit::Module model1, model2;
	model1 = torch::jit::load(argv[1], at::kCUDA);
	model2 = torch::jit::load(argv[2], at::kCUDA);
	model1.eval(); model2.eval();
	num_games = argc > 3 ? atoi(argv[3]) : 500;
	num_threads = argc > 4 ? atoi(argv[4]) : 50;

	//if (argc > 1)
	//	num_games = atoi(argv[1]);
	//if (argc > 2)
	//	num_threads = atoi(argv[2]);
	//if (argc > 3)
	//	num_report = atoi(argv[3]);
	//if (argc > 4)
	//	batch_size = atoi(argv[4]);
	//if (argc > 5)
		//save_size = atoi(argv[5]);
	//const std::string path_prefix = R"(D:\Programs\mybaguatrain\)";
	const std::string path_prefix = "";
	jit::Module* model_p = nullptr;
	jit::Module model;
	try
	{
		model = torch::jit::load(path_prefix + "latest.pt", at::kCUDA);
		model.eval();
		model_p = &model;
	}
	catch (...) {}
	std::threadpool thp(num_threads);
	std::vector<std::future<void>> futures;
	Location::policyInit();
	Board::initHash();
	std::vector<std::string> mvhist;
	/*
	//player part
	std::vector<Board> hist{ Board(9,9) };
	MCTS mcts(&modelp, search_threads, 3, num_search, .9, hist.back().get_action_size(), .085, true, true);
	while (true)
	{
		Board bd(hist.back());
		cout << bd;
		if (bd.winner != C_WALL)
		{
			auto str = hist.back().winner == C_EMPTY ? "Draw!" : hist.back().winner == P_BLACK ? "The first player wins!" : "The second player wins!";
			cout << str << endl;
		}
		//std::vector<double> action_probs(board->get_action_size(), 0);
		//std::vector<int> probs(current.get_action_size(),0);
		//probs=current.get_legal_moves();
		//{
			//std::lock_guard<std::mutex> lock(this->lock);
		std::string input;
		cin >> input;
		if (input == "printmoves")
		{
			for (const auto& s : mvhist)cout << s << endl;
			continue;
		}
		if (input != "undo" && bd.winner != C_WALL)
		{
			cout << "Game Has Ended!" << endl;
			continue;
		}
		if (input == "undo")
		{
			if (hist.size() == 1) { cout << "Nothing to undo" << endl; continue; }
			hist.pop_back();
			mvhist.pop_back();
			mcts = MCTS(&modelp, search_threads, 3, num_search, .9, hist.back().get_action_size(), .085, true, true);
			continue;
		}
		std::vector<Move> legals;
		if (!bd.genLegalMove(legals, bd.nextPla))
		{
			hist.back().winner = getOpp(bd.nextPla);
			continue;
		}
		else if (input != "genmove" && input != "analyze" && input != "analyze_raw")
		{
			Move mv = parseMove(input);
			if (mv.x == -1 || std::find(legals.begin(), legals.end(), mv) == legals.end()) { cout << "Illegal Move" << endl; continue; }
			mcts.update_with_move(bd.moveToPolicy(mv));
			bd.playMoveAssumeLegal(mv, bd.nextPla);
			mvhist.push_back(input);
		}
		else
		{
			std::vector<unsigned> _probs, visits;
			mcts.get_action_visits(&bd, _probs, visits);
			auto vv = visits;
			std::vector<float> probs(visits.size());
			double sum = std::accumulate(visits.begin(), visits.end(), 0.);
			if (abs(sum) < FLT_EPSILON)
			{
				hist.back().winner = getOpp(bd.nextPla);
				continue;
			}
			for (int i = 0; i < visits.size(); ++i)
				probs[i] = visits[i] / sum;
			sum = 0;
			for (auto& p : probs) { if (p < .01)p = 0; else p = pow(p, 1 / .3); sum += p; }
			for (auto& p : probs)p /= sum;
			std::vector<std::pair<double, Move>> pmv;
			for (int i = 0; i < legals.size(); ++i)
				pmv.push_back(std::make_pair(probs[bd.moveToPolicy(legals[i])], legals[i]));
			std::sort(pmv.begin(), pmv.end(), [](const std::pair<double, Move>& a, const std::pair<double, Move>& b) {return a.first > b.first; });

			auto policy = chooseWithProbability(probs);
			if (input == "analyze" || input == "analyze_raw")
			{
				for (const auto& [p, mv] : pmv)
				{
					if (p < DBL_EPSILON)break;
					cout << "think:" << to_string(mv) << ' ' << p << ' ' << vv[bd.moveToPolicy(mv)] << ' ' << bd.moveToPolicy(mv) << endl;
					//current.printBoard(cout, nullptr);
				}
				cout << -mcts.root->q_sa << endl;
				if (input == "analyze_raw")
				{
					cout << endl;
					auto tmp = boardToState(bd);
					std::future<NetWork::result_type> future = modelp.pushState(tmp.first, tmp.second);
					auto [action_priors, val] = future.get();
					for (Move _mv : legals)
					{
						using namespace Location;
						Move mv = _mv;
						if (bd.nextPla == P_WHITE)_mv.rot();
						//if (action_priors[movePolicy[_mv]] > 0.001)
						cout << "raw:" << bd.moveToPolicy(mv) << ' ' << to_string(mv) << ' ' << action_priors[movePolicy[_mv]] << endl;
						//current.printBoard(cout, nullptr);
					}
					cout << val << endl;
				}
			}
			Move mv = Location::policy[policy];
			if (bd.nextPla == P_WHITE)mv.rot();
			mvhist.push_back(to_string(mv));
			cout << "play " << mvhist.back() << endl;
			bd.playPolicy(policy);
			bd.checkConsistency();

			mcts.update_with_move(policy);
		}
		hist.push_back(bd);
		//cout << policy << endl;
		//cout << current.movenum << endl;
		//current.printBoard(cout, 0);
		//cout << batch_num << endl;
	}
	//cout << hist.back();

	system("pause");
	return 0;
	*/
	std::random_device rd;

	//matching part
	int scores[3]{ 0 };
	std::vector<std::future<int>> results;
	std::vector<MatchGame*> mgs;
	NetWork m1(&model1, batch_size), m2(&model2, batch_size);
	for (int i = 0; i < num_games; ++i)
	{
		MatchGame* p = nullptr;
		if (i & 1)p = new MatchGame(&m2, &m1);
		else p = new MatchGame(&m1, &m2);
		mgs.push_back(p);
		results.emplace_back(thp.commit(std::bind(&MatchGame::playUntilEnd, p)));
	}
	for (int i = 0; i < num_games; ++i)
	{
		int result = results[i].get();
		if (result && (i & 1))
			result = 3 - result;
		++scores[result];
		system("cls");
		cout << "Thread " << i << " ended!!";
		cout << "Draws: " << scores[0] << endl;
		cout << "Model 1 wins: " << scores[1] << endl;
		cout << "Model 2 wins: " << scores[2] << endl;
		delete mgs[i];
	}
	system("cls");
	cout << "Finished!!\n" << endl;
	cout << "Draws: " << scores[0] << endl;
	cout << "Model 1 wins: " << scores[1] << endl;
	cout << "Model 2 wins: " << scores[2] << endl;
	return 0;


	NetWork modelp(model_p, batch_size);
	unsigned movecnt = 0;
	auto s = clock(), s1 = clock();
	std::vector<SelfGame*> sgs;
	std::vector<Tensor> bin, over, pols, vals;
	try
	{
		std::geometric_distribution<unsigned> u(.21);
		for (int i = 0; i < num_games; ++i)
		{
			SelfGame* p = nullptr;
			sgs.push_back(p = new SelfGame(&modelp, rd(), .85, u(rd)));
			//cout << &sg.current << endl;
			//sg.current.colors[Location::getLoc(7, 6, 9)] = S_PAWN ^ S_SIDE;
			//s = clock();
			//sg.playUntilEnd();
			auto future =
				thp.commit(std::bind(&SelfGame::playUntilEnd, p));
			futures.emplace_back(std::move(future));

			if ((i + 1) % num_threads == 0)
				cout << "Thread " << i + 1 << " is started" << endl;

			//double ss=0;
			//sss += ss;
			//mvn += sg.current.movenum;
			//cout << sg.current.movenum << "moves" << ' ';
			//cout << ss / sg.current.movenum << "s per move" << endl;
			//cout << mvn << "total moves" << ' ' << sss << 's' << ' ' << sss / mvn << "s per move" << endl;
		}
		unsigned finished = 0, checkpoint = 0;
		std::vector<bool> fin(num_games, false);
		while (finished < num_games)
			for (int i = 0; i < num_games; ++i)
			{
				if (fin[i] || futures[i].wait_for(1ms) != future_status::ready)continue;
				++finished;
				fin[i] = true;
				movecnt += sgs[i]->current.movenum;
				if (finished % num_report == 0)
				{
					cout << "Thread " << finished << " is ended!!!" << endl;
					cout << (clock() - s1) / (double)CLOCKS_PER_SEC << 's' << endl;
					s1 = clock();
				}
				//cout << v[i]->current;
				if (sgs[i]->state_bin_hist.empty())continue;
				const auto& [pr, pol, val] = sgs[i]->exportTrainingData();
				bin.push_back(pr.first);
				over.push_back(pr.second);
				pols.push_back(pol);
				vals.push_back(val);
				//cout << v[i]->current;
				//cout << thp.thrCount() << endl;
				delete sgs[i];
				if (finished % save_size == 0 || finished >= num_games)
				{
					auto c = cat(bin).cpu();
					std::ofstream out;

					//out.open(path_prefix + "binary" + to_string(checkpoint) + ".pt", ios::binary | ios::out);
					cnpy::npz_save(path_prefix + "binary" + to_string(checkpoint) + ".npz", "binary", static_cast<float*>(c.data_ptr()), { (unsigned)c.size(0), (unsigned)c.size(1),9,9 });
					//out.write(c.data(), c.size());
					//out.close();

					c = cat(over).cpu();
					cnpy::npz_save(path_prefix + "overall" + to_string(checkpoint) + ".npz", "overall", static_cast<float*>(c.data_ptr()), { (unsigned)c.size(0), (unsigned)c.size(1) });

					//out.open(path_prefix + "overall" + to_string(checkpoint) + ".pt", ios::binary | ios::out);
					//out.write(c.data(), c.size());
					//out.close();

					c = cat(pols).cpu();
					cnpy::npz_save(path_prefix + "policy_target" + to_string(checkpoint) + ".npz", "policy_target", static_cast<float*>(c.data_ptr()), { (unsigned)c.size(0), (unsigned)c.size(1) });
					//out.open(path_prefix + "policy_targets" + to_string(checkpoint) + ".pt", ios::binary | ios::out);
					//out.write(c.data(), c.size());
					//out.close();

					c = cat(vals).cpu();
					cnpy::npz_save(path_prefix + "value_target" + to_string(checkpoint) + ".npz", "value_target", static_cast<float*>(c.data_ptr()), { (unsigned)c.size(0) });
					//out.open(path_prefix + "value_targets" + to_string(checkpoint) + ".pt", ios::binary | ios::out);
					//out.write(c.data(), c.size());
					//out.close();
					++checkpoint;
					bin.clear();
					over.clear();
					pols.clear();
					vals.clear();
				}
			}
		cout << ((clock() - s) / (double)CLOCKS_PER_SEC) << 's' << ' ';
		cout << "Average movenum : " << (double)movecnt / num_games << endl;
	}
	catch (const c10::Error& e)
	{
		cout << e.msg() << endl;
	}

	//cout << *a.data_ptr<float>();
	//SelfGame sg(&model, true);
	//sg.playUntilEnd();

	//Board::initHash();
	//srand(time(nullptr));
	//Location::policyInit();
	//cout << Location::policy.size() << endl;
	////board.printBoard(cout,nullptr);
	//while (true)
	//{
	//	Board board(9, 9);
	//	for (int i = 0; i < 1000; ++i)
	//	{
	//		std::vector<Move> v;
	//		board.genLegalMove(v, board.nextPla);
	//		using namespace Location;
	//		Move mm(0, 0, 0, 0);
	//		for (int j = 0; j < v.size(); ++j)
	//		{
	//			mm = v[j];
	//			if (Location::movePolicy[board.nextPla == P_BLACK ? v[j] : mm.rot()] == -1)
	//			{
	//				if (v[j].z)cout << "YES\n";
	//				cout << '(' << getX(mm.x, 9) + 1 << ',' << getY(mm.x, 9) + 1 << ")->(" << getX(mm.y, 9) + 1 << ',' << getY(mm.y, 9) + 1 << ") " << (int)mm.z <<' '<< mm.change << endl;
	//				cout << "???" << endl;
	//				board.checkConsistency();
	//				board.printBoard(cout, nullptr);
	//				return 0;
	//			}
	//		}
	//		Move mv = v[rand() % v.size()];
	//		//cout << v.size() << endl;
	//		//cout << "played:" << '(' << getX(mv.x, 9) + 1 << ',' << getY(mv.x, 9) + 1 << ")->(" << getX(mv.y, 9) + 1 << ',' << getY(mv.y, 9) + 1 << ") " << (int)mv.z << mv.change << endl;
	//		board.playMoveAssumeLegal(mv, board.nextPla);
	//		if (board.winner != C_WALL)
	//		{
	//			board.printBoard(cout, nullptr);
	//			break;
	//		}
	//	}
	//		getchar();
	//}
	//int cnt[6]{ 0 };
	//for (int i = 0; i <= 7000000; ++i)
	//	++cnt[chooseWithProbability(std::vector<double>({ 0,0.49,0.51 }))];
	//for(int i=0;i<6;++i)
	//	cout << i<<':'<<cnt[i]/ 7000000. << endl;
	//model.eval();
	//Tensor ts1 = torch::ones({ bin_num, 9, 9 }), ts2 = torch::ones({ 16 });
	//cout << torch::randn({ 4 }) << endl;
	////std::vector<torch::jit::IValue> inputs;
	//torch::jit::Stack inputs;
	//inputs.push_back(ts1.unsqueeze(0));
	//inputs.push_back(ts2.unsqueeze(0));
	//inputs.push_back(ts1);
	//inputs.push_back(ts2);
	//inputs.push_back(torch::ones({ 1, 3, 224, 224 }));
	//inputs.push_back(torch::ones({ 1, 3, 224, 224 }));

	// 执行模型并将输出转化为张量
	//try
	//{
	//	auto output = model.forward(inputs).toTuple();
	//	Tensor a= output->elements()[1].toTensor().view(-1);
	//	//a.sort()
	//	cout<<*a.data_ptr<float>();
	//}
	//catch (const c10::Error& e)
	//{
	//	cout << e.what();
	//}
	//std::vector<torch::jit::IValue> inputs;
	//inputs.push_back(ts1);
	//inputs.push_back(ts2);
	//auto x = model(std::tuple<Tensor,Tensor>(ts1,ts2)).toTensor();
	//model({})
	//Tensor t = tensor({ {1.},{2.},{3.} });
	//torch::from_blob()
	//Location::policyInit();
	//cout << Location::policy.size() << endl;
	//Board board;
	//board.printBoard(cout, nullptr);
	//std::vector<Move> mv;
	//using Location::getLoc;
	//board.printBoard(cout, nullptr);
	//board.genLegalMove(mv, board.nextPla);
	//cout << mv.size() << endl;
	//for (auto v : mv)
	//{
	//	int x = Location::getX(v.x,9), y = Location::getY(v.x,9);
	//	cout << x << ',' << y << "->";
	//	x = Location::getX(v.y, 9), y = Location::getY(v.y, 9);
	//	cout << x << ',' << y << endl;
	//}
	/*
	std::default_random_engine e(time(0));
	std::uniform_real_distribution<double> u(-1.0, 1.0);
	srand(time(0));
	auto s = Sequential(Linear(1, 16), ReLU(), Linear(16, 1));
	//s->to(at::kCUDA);
	unsigned st = clock();
	auto opt = Adam(s->parameters(), AdamOptions(0.01));
	for (int i = 0; i < 20000; ++i)
	{
		double x = u(e);
		double y = (x + 0.1) * (x + 0.1);
		auto target = torch::tensor({ y });
		auto pred = s->forward(torch::tensor({ x }));
		auto loss = functional::mse_loss(pred, target);
		if (i % 100 == 0)cout << i << "th" << endl;
		opt.zero_grad();
		loss.backward();
		opt.step();
	}
	cout << (clock() - st) / (double)CLOCKS_PER_SEC << 's' << endl;
	while (true)
	{
		double x;
		scanf_s("%lf", &x);
		auto pred = s->forward(torch::tensor({ x }));
		std::cout << pred[0] << std::endl;
	}*/
}