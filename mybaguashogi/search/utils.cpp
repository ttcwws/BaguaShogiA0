#pragma once
#include "utils.h"

#include <iostream>
#include <chrono>
std::mutex cout_mutex;

size_t MAX_BATCH_SIZE = 256;

using namespace std::chrono_literals;
void boardToStateRaw(const Board& board, std::vector<int8_t>& bin, std::vector<float>& overall)
{
    constexpr size_t bin_num = 55, overall_num = 15, board_size = 9;
    bin.assign(bin_num * board_size * board_size, 0);
    overall.assign(overall_num, 0.0f);
	auto index = [=](int k, int i, int j) {return k * board_size * board_size + i * board_size + j; };
	Player pla = board.nextPla;
	Player opp = getOpp(pla);
	for (int i = 0; i < board.x_size; ++i)
		for (int j = 0; j < board.y_size; ++j)
		{
			int x = i, y = j;
			if (pla == P_WHITE)x = board.x_size - 1 - i, y = board.y_size - 1 - j;
			Loc loc = Location::getLoc(x, y, board.x_size);
			Piece pc = board.colors[loc];
			Piece tp = getType(pc);
			if (board.isOnBoard(i, j))
				bin[index(0, i, j)] = 1;
			if (pc != C_EMPTY && pc != C_WALL)
			{
				Piece ch = tp;
				if (getPromoted(pc))
					ch += 6;
				if (getHorizontal(pc))
					ch += 13;
				if (getSide(pc) == pla)
					bin[index(ch, i, j)] = 1;
				else if (getSide(pc) == opp)
					bin[index(ch + 27, i, j)] = 1;
			}
		}
	//0 : if self king is attacked
	//1~7 : self koma
	//8~14 : opponent koma
	for (Piece i = S_KING + 1; i <= S_PAWN; ++i)
		if (i == S_PAWN)
		{
			overall[i - 1] = board.koma[pla][i] / 14. - 1;
			overall[i + 6] = board.koma[opp][i] / 14. - 1;
		}
		else if (i == S_ROOK || i == S_BISHOP)
		{
			overall[i - 1] = board.koma[pla][i] - 1;
			overall[i + 6] = board.koma[opp][i] - 1;
		}
		else
		{
			overall[i - 1] = board.koma[pla][i] / 2. - 1;
			overall[i + 6] = board.koma[opp][i] / 2. - 1;
		}
	if (GameLogic::locAttacked(board, board.kings[pla], opp))overall[0] = 1;
}
void boardToSaveRaw(const Board& board, std::vector<Piece>& color, std::vector<float>& overall)
{
	constexpr size_t overall_num = 15, board_size = 9;
	color.assign(board_size * board_size, 0);
	overall.assign(overall_num, 0.0f);
	auto index = [=](int i, int j) {return i * board_size + j; };
	Player pla = board.nextPla;
	Player opp = getOpp(pla);
	for (int i = 0; i < board.x_size; ++i)
		for (int j = 0; j < board.y_size; ++j)
		{
			int x = i, y = j;
			if (pla == P_WHITE)x = board.x_size - 1 - i, y = board.y_size - 1 - j;
			Loc loc = Location::getLoc(x, y, board.x_size);
			Piece pc = board.colors[loc];
			if (pc != C_EMPTY && pc != C_WALL)
				if (pla == P_WHITE)pc ^= S_SIDE;
			color[index(i, j)] = pc;
		}
	//0 : if self king is attacked
	//1~7 : self koma
	//8~14 : opponent koma
	for (Piece i = S_KING + 1; i <= S_PAWN; ++i)
		if (i == S_PAWN)
		{
			overall[i - 1] = board.koma[pla][i] / 14. - 1;
			overall[i + 6] = board.koma[opp][i] / 14. - 1;
		}
		else if (i == S_ROOK || i == S_BISHOP)
		{
			overall[i - 1] = board.koma[pla][i] - 1;
			overall[i + 6] = board.koma[opp][i] - 1;
		}
		else
		{
			overall[i - 1] = board.koma[pla][i] / 2. - 1;
			overall[i + 6] = board.koma[opp][i] / 2. - 1;
		}
	if (GameLogic::locAttacked(board, board.kings[pla], opp))overall[0] = 1;
}
