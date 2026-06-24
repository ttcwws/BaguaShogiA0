#include "selfplay.h"
#include "game/gamelogic.h"
#include <algorithm>
#include <cstring>

namespace
{
	struct InitialPieceSpec
	{
		Piece type;
	};

	size_t randomIndex(Rand& rand, size_t size)
	{
		return static_cast<size_t>(rand.nextUInt() % size);
	}

	bool randomTenths(Rand& rand, unsigned tenths)
	{
		return rand.nextUInt() % 10 < tenths;
	}

	Player randomPlayer(Rand& rand)
	{
		return (rand.nextUInt() & 1ULL) ? P_BLACK : P_WHITE;
	}

	template <typename T>
	void shuffleVector(std::vector<T>& values, Rand& rand)
	{
		for (size_t i = values.size(); i > 1; --i)
			std::swap(values[i - 1], values[randomIndex(rand, i)]);
	}

	std::vector<Loc> getPlayableLocs(const Board& board)
	{
		std::vector<Loc> locs;
		locs.reserve(board.x_size * board.y_size - 2);
		for (int y = 0; y < board.y_size; ++y)
			for (int x = 0; x < board.x_size; ++x)
				if (board.isOnBoard(x, y))
					locs.push_back(Location::getLoc(x, y, board.x_size));
		return locs;
	}

	const std::vector<InitialPieceSpec>& getInitialPieceSpecs()
	{
		static const std::vector<InitialPieceSpec> specs = [] {
			Board initial(9, 9);
			std::vector<InitialPieceSpec> ret;
			ret.reserve(48);
			for (int y = 0; y < initial.y_size; ++y)
				for (int x = 0; x < initial.x_size; ++x)
				{
					if (!initial.isOnBoard(x, y))
						continue;
					Piece pc = initial.colors[Location::getLoc(x, y, initial.x_size)];
					if (pc == C_EMPTY || pc == C_WALL)
						continue;
					ret.push_back({ getType(pc) });
				}
			return ret;
			}();
		return specs;
	}

	void clearBoardForShuffle(Board& board)
	{
		for (int y = 0; y < board.y_size; ++y)
			for (int x = 0; x < board.x_size; ++x)
				if (board.isOnBoard(x, y))
					board.colors[Location::getLoc(x, y, board.x_size)] = C_EMPTY;

		std::memset(board.koma, 0, sizeof board.koma);
		std::memset(board.kings, 0, sizeof board.kings);

		for (int i = 0; i < 2; ++i)
		{
			board.repetition[i].clear();
			board.first[i].clear();
		}
		for (auto& record : board.checkRecord)
		{
			record.clear();
			record.push_back(0);
		}

		board.movenum = 0;
		board.winner = C_WALL;
		board.dummy = false;
		board.attacked = false;
		board.nextPla = P_BLACK;
		board.raw_pos_hash = Hash128();
	}

	Hash128 computeRawPosHash(const Board& board)
	{
		Hash128 hash = Board::ZOBRIST_SIZE_X_HASH[board.x_size] ^ Board::ZOBRIST_SIZE_Y_HASH[board.y_size];
		for (int loc = 0; loc < Board::MAX_ARR_SIZE; ++loc)
			if (board.colors[loc] != C_WALL)
			{
				hash ^= Board::ZOBRIST_BOARD_HASH[loc][board.colors[loc]];
				hash ^= Board::ZOBRIST_BOARD_HASH[loc][C_EMPTY];
			}
		hash ^= Board::ZOBRIST_NEXTPLA_HASH[board.nextPla];
		for (int pla = P_BLACK; pla <= P_WHITE; ++pla)
			for (int pc = S_KING + 1; pc <= S_PAWN; ++pc)
				hash ^= Board::ZOBRIST_KOMA_HASH[pla][pc][board.koma[pla][pc]];
		return hash;
	}

	void finalizeInitializedBoard(Board& board, Player nextPla)
	{
		board.nextPla = nextPla;
		board.raw_pos_hash = computeRawPosHash(board);
		board.attacked = GameLogic::locAttacked(board, board.kings[getOpp(nextPla)], nextPla);

		for (int i = 0; i < 2; ++i)
		{
			board.repetition[i].clear();
			board.first[i].clear();
		}
		for (auto& record : board.checkRecord)
		{
			record.clear();
			record.push_back(0);
		}

		board.repetition[0].insert(board.raw_pos_hash.hash0);
		board.repetition[1].insert(board.raw_pos_hash.hash1);
		board.checkRecord[nextPla].push_back(board.attacked);
		int repind = board.repetition[0].count(board.raw_pos_hash.hash0) < board.repetition[1].count(board.raw_pos_hash.hash1) ? 0 : 1;
		uint64_t rephash = repind ? board.raw_pos_hash.hash1 : board.raw_pos_hash.hash0;
		board.first[repind][rephash] = board.checkRecord[nextPla].size() - 1;

	}

	int mustPromoteAt(const Board& board, Piece piece, Loc loc)
	{
		int x = Location::getX(loc, board.x_size);
		int y = Location::getY(loc, board.x_size);
		Player pla = getSide(piece);
		if (pla == P_WHITE)
		{
			x = board.x_size - 1 - x;
			y = board.y_size - 1 - y;
		}
		Piece tp = getType(piece);
		if (getPromoted(piece) || tp == S_KING || tp == S_GOLD)
			return -1;
		if (getHorizontal(piece))
		{
			if (tp == S_KNIGHT)
				return x < 2 ? 1 : 0;
			if (tp == S_PAWN || tp == S_LANCE)
				return board.isOnBoard(x - 1, y) ? 0 : 1;
			return 0;
		}
		if (tp == S_KNIGHT)
			return y < 2 ? 1 : 0;
		if (tp == S_PAWN || tp == S_LANCE)
			return board.isOnBoard(x, y - 1) ? 0 : 1;
		return 0;
	}

	Piece buildShuffledBoardPiece(const Board& board, Rand& rand, Piece type, Player side, Loc loc)
	{
		Piece piece = type;
		if (side == P_WHITE)
			piece ^= S_SIDE;
		if (type != S_KING && (rand.nextUInt() & 1ULL))
			piece ^= S_HORIZONTAL;
		if (type == S_KING || type == S_GOLD)
			return piece;
		if (mustPromoteAt(board, piece, loc) == 1)
			return piece ^ S_PROMOTED;

		bool highPromoChance =
			GameLogic::inPromotionArea(board, piece, loc, board.x_size, board.y_size) ||
			type == S_ROOK || type == S_BISHOP;
		if (randomTenths(rand, highPromoChance ? 8 : 1))
			piece ^= S_PROMOTED;
		return piece;
	}

	Piece buildShuffledBoardPiece(const Board& board, Rand& rand, Piece type, Player side, Loc loc, bool horizontal)
	{
		Piece piece = type;
		if (side == P_WHITE)
			piece ^= S_SIDE;
		if (horizontal && type != S_KING)
			piece ^= S_HORIZONTAL;
		if (type == S_KING || type == S_GOLD)
			return piece;
		if (mustPromoteAt(board, piece, loc) == 1)
			return piece ^ S_PROMOTED;

		bool highPromoChance =
			GameLogic::inPromotionArea(board, piece, loc, board.x_size, board.y_size) ||
			type == S_ROOK || type == S_BISHOP;
		if (randomTenths(rand, highPromoChance ? 8 : 1))
			piece ^= S_PROMOTED;
		return piece;
	}

	bool hasBoardKing(const Board& board, Player pla)
	{
		Loc loc = board.kings[pla];
		if (!board.isOnBoard(loc))
			return false;
		Piece piece = board.colors[loc];
		return piece != C_EMPTY && piece != C_WALL && getSide(piece) == pla && getType(piece) == S_KING;
	}

	bool canPlacePawnWithoutLineConflict(
		const bool hasPawnV[4][COMPILE_MAX_BOARD_LEN],
		const bool hasPawnH[4][COMPILE_MAX_BOARD_LEN],
		Player pla,
		bool horizontal,
		Loc loc,
		int x_size)
	{
		const int x = Location::getX(loc, x_size);
		const int y = Location::getY(loc, x_size);
		return horizontal ? !hasPawnH[pla][y] : !hasPawnV[pla][x];
	}

	void recordPlacedPawnLine(
		bool hasPawnV[4][COMPILE_MAX_BOARD_LEN],
		bool hasPawnH[4][COMPILE_MAX_BOARD_LEN],
		Piece piece,
		Loc loc,
		int x_size)
	{
		if (piece == C_EMPTY || piece == C_WALL || getType(piece) != S_PAWN || getPromoted(piece))
			return;
		const Player pla = getSide(piece);
		const int x = Location::getX(loc, x_size);
		const int y = Location::getY(loc, x_size);
		if (getHorizontal(piece))
			hasPawnH[pla][y] = true;
		else
			hasPawnV[pla][x] = true;
	}

	bool isUsableRandomizedStart(Board& board)
	{
		if (!hasBoardKing(board, P_BLACK) || !hasBoardKing(board, P_WHITE))
			return false;
		if (GameLogic::locAttacked(board, board.kings[getOpp(board.nextPla)], board.nextPla))
			return false;
		std::vector<Move> legalMoves;
		legalMoves.reserve(256);
		return board.genLegalMove(legalMoves, board.nextPla);
	}

	void applyRandomOpening(Board& board, Rand& rand)
	{
		std::vector<Move> legalMoves;
		legalMoves.reserve(256);
		while (board.winner == C_WALL)
		{
			legalMoves.clear();
			if (!board.genLegalMove(legalMoves, board.nextPla))
			{
				board.winner = getOpp(board.nextPla);
				break;
			}
			if (board.movenum >= MAX_BIG_MOVE_NUM && !board.attacked)
			{
				board.winner = C_EMPTY;
				break;
			}
			board.playMoveAssumeLegal(legalMoves[randomIndex(rand, legalMoves.size())], board.nextPla);
			if (!randomTenths(rand, 9))
				break;
		}
	}

	bool buildShuffledOpening(Board& out, Rand& rand)
	{
		const std::vector<Loc> playableLocs = getPlayableLocs(out);
		while (true)
		{
			clearBoardForShuffle(out);

			std::vector<InitialPieceSpec> pieces = getInitialPieceSpecs();
			shuffleVector(pieces, rand);
			std::vector<Loc> freeLocs = playableLocs;
			bool hasPawnV[4][COMPILE_MAX_BOARD_LEN]{ false };
			bool hasPawnH[4][COMPILE_MAX_BOARD_LEN]{ false };
			bool complete = true;

			for (const InitialPieceSpec& spec : pieces)
			{
				const Player pieceSide = randomPlayer(rand);
				const bool mustStayOnBoard = spec.type == S_KING;
				const bool canHand = !mustStayOnBoard && out.koma[pieceSide][spec.type] < MAX_KOMA_NUM - 1;
				struct PawnBoardOption
				{
					size_t freeIndex;
					bool allowVertical;
					bool allowHorizontal;
				};
				std::vector<PawnBoardOption> pawnBoardOptions;
				size_t boardChoiceCount = freeLocs.size();
				if (spec.type == S_PAWN)
				{
					pawnBoardOptions.reserve(freeLocs.size());
					for (size_t freeIndex = 0; freeIndex < freeLocs.size(); ++freeIndex)
					{
						const Loc loc = freeLocs[freeIndex];
						const bool allowVertical = canPlacePawnWithoutLineConflict(hasPawnV, hasPawnH, pieceSide, false, loc, out.x_size);
						const bool allowHorizontal = canPlacePawnWithoutLineConflict(hasPawnV, hasPawnH, pieceSide, true, loc, out.x_size);
						if (allowVertical || allowHorizontal)
							pawnBoardOptions.push_back({ freeIndex, allowVertical, allowHorizontal });
					}
					boardChoiceCount = pawnBoardOptions.size();
				}

				if (!mustStayOnBoard && canHand && rand.nextUInt() % 100 < 30)
				{
					++out.koma[pieceSide][spec.type];
					continue;
				}

				if (!boardChoiceCount)
				{
					if (canHand)
					{
						++out.koma[pieceSide][spec.type];
						continue;
					}
					complete = false;
					break;
				}

				size_t choice = randomIndex(rand, boardChoiceCount);
				size_t freeIndex = choice;
				bool forceHorizontal = false;
				bool hasForcedOrientation = false;
				if (spec.type == S_PAWN)
				{
					const PawnBoardOption& option = pawnBoardOptions[choice];
					freeIndex = option.freeIndex;
					hasForcedOrientation = true;
					if (option.allowVertical != option.allowHorizontal)
						forceHorizontal = option.allowHorizontal;
					else
						forceHorizontal = (rand.nextUInt() & 1ULL) != 0;
				}

				Loc loc = freeLocs[freeIndex];
				freeLocs[freeIndex] = freeLocs.back();
				freeLocs.pop_back();
				Piece piece = hasForcedOrientation ? buildShuffledBoardPiece(out, rand, spec.type, pieceSide, loc, forceHorizontal) : buildShuffledBoardPiece(out, rand, spec.type, pieceSide, loc);
				out.colors[loc] = piece;
				recordPlacedPawnLine(hasPawnV, hasPawnH, piece, loc, out.x_size);
				if (spec.type == S_KING)
					out.kings[pieceSide] = loc;
			}
			if (!complete)
				continue;

			Player nextPla = randomPlayer(rand);
			finalizeInitializedBoard(out, nextPla);
			if (!isUsableRandomizedStart(out))
			{
				finalizeInitializedBoard(out, getOpp(nextPla));
				if (!isUsableRandomizedStart(out))
					continue;
			}

#ifdef _DEBUG
			out.checkConsistency();
#endif
			return true;
		}
	}

	void initializeSelfplayBoard(Board& board, Rand& rand)
	{
		unsigned roll = static_cast<unsigned>(rand.nextUInt() % 100);
		if (roll < 13)
		{
			applyRandomOpening(board, rand);
#ifdef _DEBUG
			if (board.winner == C_WALL)
				board.checkConsistency();
#endif
			cout << board;
			return;
		}
		if (roll < 20)
			buildShuffledOpening(board, rand), cout << board;
	}
}

SelfGame::SelfGame(NetWork* model, long long seed, double ratio, unsigned start, bool gpu) :
	current(9, 9), ended(false), batch_num(0), model(model), use_gpu(gpu), rand(seed), record_ratio(1), start(start), copied(false) {
}
void SelfGame::playUntilEnd()
{
	//cout << &current.repetition[0] << ' ' << &current.repetition[1] << endl;
	//current.repetition[0][1] = 1;
	float temperature = 1.2f;
	initializeSelfplayBoard(current, rand);
	MCTS mcts(model, 3, 303, POLICY_SIZE, 0.08f, true, use_gpu);
	std::vector<float> probs(POLICY_SIZE), policy(POLICY_SIZE);
	if (!model->model)mcts.num_mcts_sims = 600;
	//int start = 0;
	if (ended)return;
	std::vector<Player> pla_hist;
	pla_hist.reserve(128);
	std::vector<unsigned> visits, visits_pruned;
	std::vector<Move> mvs;
	mvs.reserve(256);
	std::vector<Piece> x;
	x.reserve(9 * 9);
	std::vector<float> y;
	y.reserve(OVERALL_SIZE);
	while (current.winner == C_WALL)
	{
		mvs.clear();
		if (!current.genLegalMove(mvs, current.nextPla))
		{
			current.winner = getOpp(current.nextPla);
			break;
		}
		else if (current.movenum >= MAX_BIG_MOVE_NUM && !current.attacked)
		{
			current.winner = C_EMPTY;
			break;
		}
		if (current.movenum >= 25 && temperature > 1.f / 30)temperature *= 0.99f;
		mcts.get_action_visits(current, visits, visits_pruned);
		unsigned sum1 = std::reduce(std::execution::par_unseq, visits.cbegin(), visits.cend());
		if (!sum1) { cout << current << endl; cout << mvs.size() << endl; throw std::runtime_error("No Valid Moves"); }

		std::transform(std::execution::par_unseq, visits.cbegin(), visits.cend(), probs.begin(), [sum1, temperature](const unsigned& v) { return pow((float)v / sum1, 1 / temperature); });
		float sum = std::reduce(std::execution::par_unseq, probs.cbegin(), probs.cend());
		std::transform(std::execution::par_unseq, probs.cbegin(), probs.cend(), probs.begin(), [sum](const float& v) {return v / sum; });
		auto action = chooseWithProbability(probs);
		unsigned sum_p = std::reduce(std::execution::par_unseq, visits_pruned.cbegin(), visits_pruned.cend());
		std::transform(std::execution::par_unseq, visits_pruned.cbegin(), visits_pruned.cend(), policy.begin(), [sum_p](const unsigned& v) { return (float)v / sum_p; });
		pla_hist.push_back(current.nextPla);
		append(policy_hist, policy);
		boardToSaveRaw(current, x, y);
		append(state_color_hist, x);
		append(state_overall_hist, y);
		++batch_num;
		current.playPolicy(action);
		mcts.update_with_move(action);
	}
	value_hist.resize(pla_hist.size());
	std::transform(pla_hist.cbegin(), pla_hist.cend(), value_hist.begin(),
		[this](const Player& pla) {
			if (current.winner == C_EMPTY)return 0;
			return current.winner == pla ? 1 : -1;
		});
	ended = true;
}
void SelfGame::exportTrainingData(
	std::vector<Piece>& color,
	std::vector<float>& overall,
	std::vector<float>& policy,
	std::vector<int8_t>& value)
{
	if (copied)
	{
		cout << "Warning: exportTrainingData called multiple times on the same SelfGame instance. Only the first call will have an effect." << endl;
		return;
	}
	// 每个状态的 board 保存为 9*9 的 Piece 顺序数组
	const int board_size = 9;
	const size_t state_count = state_color_hist.size() / (board_size * board_size);
	// 每个 state 对应的 overall 长度为 OVERALL_SIZE（15），policy 为 POLICY_SIZE（4790），value 为 1
	const size_t overall_stride = OVERALL_SIZE;
	const size_t policy_stride = POLICY_SIZE;

	for (size_t s = 0; s < state_count; ++s)
	{
		auto base_color_it = state_color_hist.begin() + s * board_size * board_size;
		std::vector<Piece> base_color(base_color_it, base_color_it + board_size * board_size);

		// 找出满足规则的飞/角位置索引
		std::vector<int> eligible_idxs;
		for (int i = 0; i < board_size; ++i)
		{
			for (int j = 0; j < board_size; ++j)
			{
				int idx = i * board_size + j;
				Piece pc = base_color[idx];
				if (pc == C_EMPTY || pc == C_WALL) continue;
				Piece tp = getType(pc);
				if (tp != S_ROOK && tp != S_BISHOP) continue;

				bool promoted = getPromoted(pc);
				// 规则2：非成的飞/角若位于敌阵角落区域之一，则也可旋转不变
				bool in_corner_region = (getSide(pc) == P_BLACK && i < 3 && j < 3) || (getSide(pc) == P_WHITE && i > 5 && j > 5);

				if (promoted || in_corner_region)
					eligible_idxs.push_back(idx);
			}
		}

		// 生成 2^n 个变体（包含原始），对每个 eligible 位置按位翻转 S_HORIZONTAL
		auto overall_begin = state_overall_hist.begin() + s * overall_stride;
		std::vector<float> overall_slice(overall_begin, overall_begin + overall_stride);

		auto policy_begin = policy_hist.begin() + s * policy_stride;
		std::vector<float> policy_slice(policy_begin, policy_begin + policy_stride);
		size_t n = eligible_idxs.size();
		size_t variants = n ? (1u << n) : 1u;
		for (size_t mask = 0; mask < variants; ++mask)
		{
			std::vector<Piece> aug_color = base_color;
			for (size_t k = 0; k < n; ++k)
			{
				if (mask & (1u << k))
				{
					int pidx = eligible_idxs[k];
					aug_color[pidx] ^= S_HORIZONTAL;
				}
			}

			// 追加颜色数据
			append(color, aug_color);

			// 追加 overall, policy, value 对应片段
			append(overall, overall_slice);
			append(policy, policy_slice);

			value.push_back(value_hist[s]);

			// // 调试输出：在增强时加锁并打印增强后的棋盘
			// if (n)
			// {
			// 	std::lock_guard<std::mutex> lk(cout_mutex);
			// 	std::cout << "[Augment] state=" << s << " mask=" << mask << "\n";
			// 	current.printBoardOnly(std::cout, &aug_color);
			// 	getchar();
			// }
		}
	}

	copied = true;
}
