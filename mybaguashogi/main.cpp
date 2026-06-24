#include <cstdio>
#include <iostream>
#include <random>
#include <ctime>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
//#include <torch/nn.h>
#include "game/board.h"
#include "search/thread_pool.h"
#include "match.h"
#include "core/rand.h"
#include "core/stable_time.h"
#include "selfplay.h"
#include "zlib.h"
//using namespace torch;
//using namespace torch::optim;
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

// 压缩并保存单个 vector

int main(int argc, char* argv[])
{
	int num_games = 2000, num_threads = 1, num_report = 50, save_size = 1050, start_checkpoint = 0;
	int num_search = 1000, search_threads = 1;
	const std::string path_prefix = "./";
	//cudaStreamCreate(&stream2);
	//const std::string path_prefix = "D:/Programs/mybaguatrain/";
	std::string onnx = "";
	std::string model1_path = "", model2_path = "";
	////player
	//if (argc > 1)
	//	num_search = atoi(argv[1]);
	//if (argc > 2)
	//	search_threads = atoi(argv[2]);

	//match
	//jit::Module model1, model2;
	//model1 = torch::jit::load(argv[1], at::kCUDA);
	//model2 = torch::jit::load(argv[2], at::kCUDA);
	//model1.eval(); model2.eval();
	//num_games = argc > 3 ? atoi(argv[3]) : 500;
	//num_threads = argc > 4 ? atoi(argv[4]) : 50;
	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];

		if (arg.substr(0, 2) == "--") {  // 长参数 --xxx
			std::string key = arg.substr(2);
			if (i + 1 < argc && std::string(argv[i + 1]).substr(0, 2) != "--") {
				if (key == "num_games")num_games = atoi(argv[++i]);
				else if (key == "num_threads")num_threads = atoi(argv[++i]);
				else if (key == "num_report")num_report = atoi(argv[++i]);
				else if (key == "batch_size")MAX_BATCH_SIZE = atoi(argv[++i]);
				else if (key == "save_size")save_size = atoi(argv[++i]);
				else if (key == "start_checkpoint")start_checkpoint = atoi(argv[++i]);
				else if (key == "num_search")num_search = atoi(argv[++i]);
				else if (key == "search_threads")search_threads = atoi(argv[++i]);
				else if (key == "model")
				{
					onnx = argv[++i];
					if (!std::filesystem::exists(onnx))
					{
						cout << "Model file not found: " << onnx << endl;
						return 0;
					}
				}
				else if (key == "model1")
				{
					model1_path = argv[++i];
					if (!std::filesystem::exists(model1_path))
					{
						cout << "Model file not found: " << model1_path << endl;
						return 0;
					}
				}
				else if (key == "model2")
				{
					model2_path = argv[++i];
					if (!std::filesystem::exists(model2_path))
					{
						cout << "Model file not found: " << model2_path << endl;
						return 0;
					}
				}
				else {
					cout << "Unrecognized argument --" << key << endl;
					return 0;
				}
			}
			else {
			}
		}
		else
		{
			cout << "Unrecognized argument " << arg << endl;
			return 0;
		}
	}
#if defined(SELFPLAY) || defined(MATCH)
	cout << "num_games: " << num_games << endl;
	cout << "num_threads: " << num_threads << endl;
#elif defined(PLAYER)
	cout << "num_search: " << num_search << endl;
#endif

#if defined(SELFPLAY) || defined(PLAYER)
	std::unique_ptr<TensorRTEngine> model_p = nullptr;
	if (!onnx.empty())
		model_p.reset(new TensorRTEngine(onnx));
#endif

	//std::vector<int8_t> b(BINARY_SIZE*2, 1);
	//std::vector<float> o(OVERALL_SIZE*2, 0.), policy, value, probs;
	//model_p->to_buffer(b, o);
	//model_p->infer(policy, value);
	//softmax(policy.begin() + POLICY_SIZE, probs);
	//for (int i = 0; i < POLICY_SIZE; ++i)
	//	if (probs[i] > .01)
	//	{
	//		cout << i << ' ' << policy[i] << ' ' << probs[i] << endl;
	//	}
	//cout << value[1] << endl;
	//return 0;
	ThreadPool thp(num_threads);
	std::vector<std::future<void>> futures;
	Location::policyInit();
	//ofstream fout("validPolicy.txt");
	//cout << "Valid Policy Size: " << Location::validPolicy.size() << endl;
	//for (const auto& a : Location::validPolicy)
	//{
	//	fout << std::get<0>(a) << ' ' << std::get<1>(a) << ' ' << std::get<2>(a) << endl;
	//}
	//return 0;
	Board::initHash();
	std::vector<std::string> mvhist;
	//NetWork modelp(model_p.get(), batch_size, 1.f);
	//player part
#ifdef PLAYER
	NetWork modelp(model_p.get(), 1.f);
	std::vector<Board> hist{ Board(9,9) };
	MCTS mcts(&modelp, 3, num_search, POLICY_SIZE, 0.08, true, true, false);
	auto parse_visit_count = [](const std::string& token, unsigned& visit_count) {
		size_t parsed = 0;
		unsigned long long value = std::stoull(token, &parsed);
		if (parsed != token.size() || value == 0 || value > std::numeric_limits<unsigned>::max())
			return false;
		visit_count = static_cast<unsigned>(value);
		return true;
	};
	while (true)
	{
		Board bd(hist.back());
		cout << bd;
		if (bd.winner != C_WALL)
		{
			auto str = hist.back().winner == C_EMPTY ? "Draw!" : hist.back().winner == P_BLACK ? "The first player wins!" : "The second player wins!";
			cout << str << endl;
		}
		std::string input_line;
		std::getline(std::cin >> std::ws, input_line);
		std::istringstream input_stream(input_line);
		std::vector<std::string> tokens;
		for (std::string token; input_stream >> token;)
			tokens.push_back(token);
		if (tokens.empty())
			continue;
		std::string input = tokens[0];
		unsigned visit_count = static_cast<unsigned>(num_search);
		const bool supports_visit_count = input == "simulate" || input == "analyze" || input == "analyze_raw";
		if (supports_visit_count)
		{
			if (tokens.size() > 2)
			{
				cout << "Usage: " << input << " [visit_count]" << endl;
				continue;
			}
			if (tokens.size() == 2)
			{
				try
				{
					if (!parse_visit_count(tokens[1], visit_count))
					{
						cout << "visit_count must be a positive integer" << endl;
						continue;
					}
				}
				catch (const std::exception&)
				{
					cout << "visit_count must be a positive integer" << endl;
					continue;
				}
			}
		}
		else if (tokens.size() != 1)
		{
			cout << "Unexpected extra arguments" << endl;
			continue;
		}
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
			mcts = MCTS(&modelp, 3, num_search, POLICY_SIZE, 0.08, true, true, false);
			continue;
		}
		std::vector<Move> legals;
		if (!bd.genLegalMove(legals, bd.nextPla))
		{
			hist.back().winner = getOpp(bd.nextPla);
			continue;
		}
		else if (input != "genmove" && input != "analyze" && input != "analyze_raw" && input != "simulate")
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
			const unsigned original_num_mcts_sims = mcts.num_mcts_sims;
			mcts.num_mcts_sims = visit_count;
			mcts.get_action_visits(bd, _probs, visits);
			mcts.num_mcts_sims = original_num_mcts_sims;
			auto vv = visits;
			std::vector<float> probs(visits.size());
			float sum = std::accumulate(visits.begin(), visits.end(), 0.f);
			if (abs(sum) < FLT_EPSILON)
			{
				hist.back().winner = getOpp(bd.nextPla);
				continue;
			}
			for (int i = 0; i < visits.size(); ++i)
				probs[i] = visits[i] / sum;
			sum = 0;
			for (auto& p : probs) { if (p < FLT_EPSILON)p = 0; else p = pow(p, 3); sum += p; }
			for (auto& p : probs)p /= sum;
			std::vector<std::pair<float, Move>> pmv;
			for (int i = 0; i < legals.size(); ++i)
				pmv.push_back(std::make_pair(probs[bd.moveToPolicy(legals[i])], legals[i]));
			std::sort(pmv.begin(), pmv.end(), [](const std::pair<float, Move>& a, const std::pair<float, Move>& b) {return a.first > b.first; });

			auto policy = chooseWithProbability(probs);
			if (input == "analyze" || input == "analyze_raw" || input == "simulate")
			{
				for (const auto& [p, mv] : pmv)
				{
					if (!vv[bd.moveToPolicy(mv)])break;
					cout << "think:" << to_string(mv) << ' ' << p << ' ' << vv[bd.moveToPolicy(mv)] << ' ' << bd.moveToPolicy(mv) << endl;
					//current.printBoard(cout, nullptr);
				}
				cout << mcts.root->get_value() << endl;
				if (input == "analyze_raw")
				{
					cout << endl;
					auto [action_priors, val] = modelp.predict(bd);
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
			if (input != "simulate")
			{
				Move mv = Location::policy[policy];
				if (bd.nextPla == P_WHITE)mv.rot();
				mvhist.push_back(to_string(mv));
				cout << "play " << mvhist.back() << endl;
				bd.playPolicy(policy);
				bd.checkConsistency();

				mcts.update_with_move(policy);
			}
		}
		if (input != "simulate")
			hist.push_back(bd);
		//cout << policy << endl;
		//cout << current.movenum << endl;
		//current.printBoard(cout, 0);
		//cout << batch_num << endl;
	}
	//cout << hist.back();

	system("pause");
	return 0;
#endif
	std::random_device rd;
	std::vector<bool> fin(num_games, false);
	unsigned finished = 0;
	//matching part
#ifdef MATCH
	TensorRTEngine* t1 = nullptr, *t2 = nullptr;
	if (model1_path.empty() || model2_path.empty())
	{
		cout << "Using random." << endl;
	}
	if (!model1_path.empty())
		t1 = new TensorRTEngine(model1_path);
	if (!model2_path.empty())
		t2 = new TensorRTEngine(model2_path);
	NetWork m1(t1, 1.f), m2(t2, 1.f);
	int scores[3]{ 0 };
	std::vector<std::future<int>> results;
	std::vector<MatchGame*> mgs;
	for (int i = 0; i < num_games; ++i)
	{
		MatchGame* p = nullptr;
		if (i & 1)p = new MatchGame(&m2, &m1, 500);
		else p = new MatchGame(&m1, &m2, 500);
		mgs.push_back(p);
		results.emplace_back(thp.commit(std::bind(&MatchGame::playUntilEnd, p)));
	}
	while (finished < num_games)
	{
		bool progress = false;
		for (int i = 0; i < num_games; ++i)
		{
			if (fin[i])continue;
			if (results[i].wait_for(1ms) == std::future_status::ready)
			{
				fin[i] = true;
				progress = true;
				finished++;
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
		}
		if (!progress)
			std::this_thread::sleep_for(500ms);
	}
	system("cls");
	cout << "Finished!!\n" << endl;
	cout << "Draws: " << scores[0] << endl;
	cout << "Model 1 wins: " << scores[1] << endl;
	cout << "Model 2 wins: " << scores[2] << endl;
	//cudaStreamDestroy(stream1);
	return 0;
#endif

#ifdef SELFPLAY
	NetWork modelp(model_p.get(), 1.f);
	unsigned movecnt = 0;
	auto s = stable_time(), s1 = stable_time();
	std::vector<SelfGame*> sgs;
	std::vector<Piece> color;
	std::vector<int8_t> vals;
	std::vector<float> over, pols;
	try
	{
		std::geometric_distribution<unsigned> u(.21);
		for (int i = 0; i < num_games; ++i)
		{
			SelfGame* p = nullptr;
			sgs.push_back(p = new SelfGame(&modelp, rd(), 1, u(rd)));

			auto future =
				thp.commit(std::bind(&SelfGame::playUntilEnd, p));
			futures.emplace_back(std::move(future));
		}
		unsigned finished = 0, checkpoint = start_checkpoint;
		

		using namespace std::chrono_literals;
		while (finished < (unsigned)num_games)
		{
			bool progress = false;
			for (int i = 0; i < num_games; ++i)
			{
				if (fin[i]) continue;

				// 非阻塞检查该 future 是否已完成
				if (futures[i].wait_for(1ms) == std::future_status::ready)
				{
					// 获取结果/重新抛出异常（如果 playUntilEnd 抛出）
					try
					{
						futures[i].get();
					}
					catch (const std::exception& e)
					{
						std::cerr << "SelfGame thread exception: " << e.what() << std::endl;
					}
					catch (...)
					{
						std::cerr << "SelfGame thread unknown exception" << std::endl;
					}

					fin[i] = true;
					++finished;
					progress = true;

					// 累加统计并导出该局训练数据（若有）
					movecnt += sgs[i]->current.movenum;
					if (finished % num_report == 0)
					{
						std::cout << "Thread " << finished << " has ended!!!" << std::endl;
						std::cout << to_seconds_float(stable_time() - s1) << 's' << std::endl;
						s1 = stable_time();
					}

					if (!sgs[i]->state_color_hist.empty())
						sgs[i]->exportTrainingData(color, over, pols, vals);

					delete sgs[i];
					sgs[i] = nullptr;

					// 周期性保存（压缩写盘）
					if (finished % save_size == 0 || finished >= (unsigned)num_games)
					{
						auto s = color.size() / (9 * 9);
						if (over.size() / 15 != s || pols.size() / 4790 != s || vals.size() != s)
						{
							cout << "color size: " << s << endl;
							cout << "over size: " << over.size() / 15 << endl;
							cout << "pols size: " << pols.size() / 4790 << endl;
							cout << "vals size: " << vals.size() << endl;
							throw std::runtime_error("Data size mismatch!");
						}
						std::filesystem::create_directory(path_prefix + "training_data/" + std::to_string(checkpoint));
						save_vector_zlib(path_prefix + "training_data/" + std::to_string(checkpoint) + "/color.ttc", color);
						save_vector_zlib(path_prefix + "training_data/" + std::to_string(checkpoint) + "/over.ttc", over);
						save_vector_zlib(path_prefix + "training_data/" + std::to_string(checkpoint) + "/pols.ttc", pols);
						save_vector_zlib(path_prefix + "training_data/" + std::to_string(checkpoint) + "/vals.ttc", vals);
						++checkpoint;
						color.clear();
						over.clear();
						pols.clear();
						vals.clear();
					}
				}
			}

			// 如果这一轮没有任何进展，短暂睡眠避免 busy-wait
			if (!progress)
				std::this_thread::sleep_for(5ms);
		}
		cout << to_seconds_float(stable_time() - s) << 's' << ' ';
		cout << "Average movenum : " << (float)movecnt / num_games << endl;
	}
	catch (const std::exception& e)
	{
		cout << e.what() << endl;
	}
#endif
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
	auto st = stable_time();
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
	cout << to_seconds_double(stable_time() - st) << 's' << endl;
	while (true)
	{
		double x;
		scanf_s("%lf", &x);
		auto pred = s->forward(torch::tensor({ x }));
		std::cout << pred[0] << std::endl;
	}*/
}
