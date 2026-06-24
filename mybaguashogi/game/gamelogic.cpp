#include "../game/gamelogic.h"

/*
 * gamelogic.cpp
 * Logics of game rules
 * Some other game logics are in board.h/cpp
 *
 * Gomoku as a representive
 */

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

using namespace std;
void getdXY(int adj, int x_size, int& x, int& y) {
	int absadj = abs(adj);
	x = absadj % (x_size + 1);
	x = x > 4 ? x - x_size - 1 : x;
	y = (absadj - x) / (x_size + 1);
	if (adj < 0)
		x = -x, y = -y;
}
int getRotatedOffset(int adj, int x_size, int count = 1) {
	if ((abs(count) & 1) == 0)
		return abs(count) % 4 == 0 ? adj : -adj;
	for (int i = 0; i < abs(count); ++i) {
		int x, y;
		getdXY(adj, x_size, x, y);
		swap(x, y);
		if (count > 0)
			x = -x;
		else
			y = -y;
		adj = y * (x_size + 1) + x;
	}
	return adj;
}

bool GameLogic::inPromotionArea(const Board& board, Piece piece, Loc loc, int x_size, int y_size) {
	int x = Location::getX(loc, x_size);
	int y = Location::getY(loc, x_size);
	int pla = getSide(piece);
	if (getPromoted(piece) || getType(piece) == S_GOLD || getType(piece) == S_KING || piece == C_EMPTY)
		return false;
	if (!board.isOnBoard(loc))
		return false;
	if (pla == P_WHITE)
		x = x_size - 1 - x, y = y_size - 1 - y;
	return getHorizontal(piece) ? 0 <= x && x < 3 : y_size - 3 <= y && y < y_size;
}
bool GameLogic::inKnightMustPromoteArea(const Board& board, Piece piece, Loc loc) {
	const int& x_size = board.x_size, & y_size = board.y_size;
	int x = Location::getX(loc, x_size);
	int y = Location::getY(loc, x_size);
	int pla = getSide(piece);
	if (getPromoted(piece) || getType(piece) != S_KNIGHT)
		return false;
	if (pla == P_WHITE)
		x = x_size - 1 - x, y = y_size - 1 - y;
	return getHorizontal(piece) ? 0 <= x && x < 2 : y_size - 2 <= y && y < y_size;
}
bool GameLogic::getAttackingPiece(const Board& board, Loc loc, Player pla, vector<Loc>& res) {
	static constexpr int rotInd[]{ 1,3,0,2,6,4,7,5 };
	static constexpr int gold[]{ 0,1,2,3,4,5 }, silver[]{ 0,4,5,6,7 }, pawn[]{ 0 };
	bool mark[9 * 9]{ false };
	int x = Location::getX(loc, board.x_size), y = Location::getY(loc, board.x_size);
	//check gold
	auto isGoldMove = [](const Piece& p) {
		Piece tp = getType(p);
		return tp == S_GOLD || getPromoted(p) && (tp == S_KNIGHT || tp == S_LANCE || tp == S_PAWN || tp == S_SILVER);
		};
	for (int i : gold)
	{
		int nx = x, ny = y, j = rotInd[i];
		if (pla == P_WHITE)nx += board.adj_ofx[i], ny += board.adj_ofy[i];
		else nx -= board.adj_ofx[i], ny -= board.adj_ofy[i];
		Loc nl = Location::getLoc(nx, ny, board.x_size);
		if (board.isOnBoard(nx, ny))
		{
			Piece pc = board.colors[nl];
			if (pc != C_EMPTY && getSide(pc) == pla)
				if (!getHorizontal(pc) && isGoldMove(pc))mark[nl] = true;
		}
		nx = x; ny = y;
		if (pla == P_WHITE)nx += board.adj_ofx[j], ny += board.adj_ofy[j];
		else nx -= board.adj_ofx[j], ny -= board.adj_ofy[j];
		nl = Location::getLoc(nx, ny, board.x_size);
		if (board.isOnBoard(nx, ny))
		{
			Piece pc = board.colors[nl];
			if (pc != C_EMPTY && getSide(pc) == pla)
				if (getHorizontal(pc) && isGoldMove(pc))mark[nl] = true;
		}
	}
	//check silver
	for (int i : silver)
	{
		int nx = x, ny = y, j = rotInd[i];
		if (pla == P_WHITE)nx += board.adj_ofx[i], ny += board.adj_ofy[i];
		else nx -= board.adj_ofx[i], ny -= board.adj_ofy[i];
		Loc nl = Location::getLoc(nx, ny, board.x_size);
		if (board.isOnBoard(nx, ny))
		{
			Piece pc = board.colors[nl];
			if (pc != C_EMPTY && getSide(pc) == pla)
				if (!getPromoted(pc) && !getHorizontal(pc) && getType(pc) == S_SILVER)mark[nl] = true;
		}
		nx = x; ny = y;
		if (pla == P_WHITE)nx += board.adj_ofx[j], ny += board.adj_ofy[j];
		else nx -= board.adj_ofx[j], ny -= board.adj_ofy[j];
		nl = Location::getLoc(nx, ny, board.x_size);
		if (board.isOnBoard(nx, ny))
		{
			Piece pc = board.colors[nl];
			if (pc != C_EMPTY && getSide(pc) == pla)
				if (!getPromoted(pc) && getHorizontal(pc) && getType(pc) == S_SILVER)mark[nl] = true;
		}
	}
	//check King, Promoted Bishop and Promoted Rook
	for (int i = 0; i < 8; ++i)
	{
		int nx = x, ny = y;
		nx += board.adj_ofx[i], ny += board.adj_ofy[i];
		Loc nl = Location::getLoc(nx, ny, board.x_size);
		if (board.isOnBoard(nx, ny))
		{
			Piece pc = board.colors[nl];
			Piece tp = getType(pc);
			if (pc != C_EMPTY && getSide(pc) == pla)
				if (tp == S_KING || (getPromoted(pc) && (tp == S_BISHOP || tp == S_ROOK)))mark[nl] = true;
		}
	}
	//check Knight
	int k_ofx[]{ -1,1,-2,-2 }, k_ofy[]{ -2,-2,-1,1 };
	for (int i = 0; i < 2; ++i)
	{
		int nx = x, ny = y;
		if (pla == P_WHITE)nx += k_ofx[i], ny += k_ofy[i];
		else nx -= k_ofx[i], ny -= k_ofy[i];
		Loc nl = Location::getLoc(nx, ny, board.x_size);
		if (board.isOnBoard(nx, ny))
		{
			Piece pc = board.colors[nl];
			Piece tp = getType(pc);
			if (pc != C_EMPTY && getSide(pc) == pla)
				if (!getPromoted(pc) && !getHorizontal(pc) && tp == S_KNIGHT)mark[nl] = true;
		}
	}
	for (int i = 2; i < 4; ++i)
	{
		int nx = x, ny = y;
		if (pla == P_WHITE)nx += k_ofx[i], ny += k_ofy[i];
		else nx -= k_ofx[i], ny -= k_ofy[i];
		Loc nl = Location::getLoc(nx, ny, board.x_size);
		if (board.isOnBoard(nx, ny))
		{
			Piece pc = board.colors[nl];
			Piece tp = getType(pc);
			if (pc != C_EMPTY && getSide(pc) == pla)
				if (!getPromoted(pc) && getHorizontal(pc) && tp == S_KNIGHT)mark[nl] = true;
		}
	}
	//check Pawn and Lance

	for (int i : pawn)
	{
		int nx = x, ny = y, j = rotInd[i];
		int ofx = board.adj_ofx[i], ofy = board.adj_ofy[i];
		if (pla != P_WHITE)ofx = -ofx, ofy = -ofy;
		nx += ofx; ny += ofy;
		Loc nl = Location::getLoc(nx, ny, board.x_size);
		if (board.isOnBoard(nx, ny))
		{
			Piece pc = board.colors[nl];
			Piece tp = getType(pc);
			if (pc != C_EMPTY && getSide(pc) == pla)
				if (!getHorizontal(pc) && !getPromoted(pc) && (tp == S_LANCE || tp == S_PAWN))mark[nl] = true;
		}
		if (board.colors[nl] == C_EMPTY)while (true)
		{
			nx += ofx; ny += ofy;
			Loc nl = Location::getLoc(nx, ny, board.x_size);
			if (!board.isOnBoard(nx, ny))break;
			Piece pc = board.colors[nl];
			if (pc == C_EMPTY)continue;
			Piece tp = getType(pc);
			if (getSide(pc) == pla)
				if (!getHorizontal(pc) && !getPromoted(pc) && tp == S_LANCE)mark[nl] = true;
			break;
		}

		nx = x; ny = y;
		ofx = board.adj_ofx[j], ofy = board.adj_ofy[j];
		if (pla != P_WHITE)ofx = -ofx, ofy = -ofy;
		nx += ofx; ny += ofy;
		nl = Location::getLoc(nx, ny, board.x_size);
		if (board.isOnBoard(nx, ny))
		{
			Piece pc = board.colors[nl];
			Piece tp = getType(pc);
			if (pc != C_EMPTY && getSide(pc) == pla)
				if (getHorizontal(pc) && !getPromoted(pc) && (tp == S_LANCE || tp == S_PAWN))mark[nl] = true;
		}
		if (board.colors[nl] == C_EMPTY)while (true)
		{
			nx += ofx; ny += ofy;
			Loc nl = Location::getLoc(nx, ny, board.x_size);
			if (!board.isOnBoard(nx, ny))break;
			Piece pc = board.colors[nl];
			if (pc == C_EMPTY)continue;
			Piece tp = getType(pc);
			if (getSide(pc) == pla)
				if (getHorizontal(pc) && !getPromoted(pc) && tp == S_LANCE)mark[nl] = true;
			break;
		}
	}
	//check Rook
	for (int i = 0; i < 4; ++i)
	{
		int nx = x, ny = y;
		int ofx = board.adj_ofx[i], ofy = board.adj_ofy[i];
		if (pla != P_WHITE)ofx = -ofx, ofy = -ofy;
		nx += ofx; ny += ofy;
		while (board.isOnBoard(nx, ny))
		{
			Loc nl = Location::getLoc(nx, ny, board.x_size);
			Piece pc = board.colors[nl];
			nx += ofx; ny += ofy;

			if (pc == C_EMPTY)continue;
			Piece tp = getType(pc);
			if (getSide(pc) == pla)
				if (tp == S_ROOK)mark[nl] = true;
			break;
		}
	}
	//check Bishop
	for (int i = 4; i < 8; ++i)
	{
		int nx = x, ny = y;
		int ofx = board.adj_ofx[i], ofy = board.adj_ofy[i];
		if (pla != P_WHITE)ofx = -ofx, ofy = -ofy;
		nx += ofx; ny += ofy;
		while (board.isOnBoard(nx, ny))
		{
			Loc nl = Location::getLoc(nx, ny, board.x_size);
			Piece pc = board.colors[nl];
			nx += ofx; ny += ofy;
			if (pc == C_EMPTY)continue;
			Piece tp = getType(pc);
			if (getSide(pc) == pla)
				if (tp == S_BISHOP)mark[nl] = true;
			break;
		}
	}
	for (int i = 0; i < 81; ++i)
		if (mark[i])res.push_back(i);
	return !res.empty();
}
bool GameLogic::locAttacked(const Board& board, Loc loc, Player pla) {
	static constexpr int rotInd[]{ 1,3,0,2,6,4,7,5 };
	static constexpr int gold[]{ 0,1,2,3,4,5 }, silver[]{ 0,4,5,6,7 }, pawn[]{ 0 };
	int x = Location::getX(loc, board.x_size), y = Location::getY(loc, board.x_size);
	//check gold
	auto isGoldMove = [](const Piece& p) {
		Piece tp = getType(p);
		return tp == S_GOLD || getPromoted(p) && (tp == S_KNIGHT || tp == S_LANCE || tp == S_PAWN || tp == S_SILVER);
		};
	for (int i : gold)
	{
		int nx = x, ny = y, j = rotInd[i];
		if (pla == P_WHITE)nx += board.adj_ofx[i], ny += board.adj_ofy[i];
		else nx -= board.adj_ofx[i], ny -= board.adj_ofy[i];
		Loc nl = Location::getLoc(nx, ny, board.x_size);
		if (board.isOnBoard(nx, ny))
		{
			Piece pc = board.colors[nl];
			if (pc != C_EMPTY && getSide(pc) == pla)
				if (!getHorizontal(pc) && isGoldMove(pc))return true;
		}
		nx = x; ny = y;
		if (pla == P_WHITE)nx += board.adj_ofx[j], ny += board.adj_ofy[j];
		else nx -= board.adj_ofx[j], ny -= board.adj_ofy[j];
		nl = Location::getLoc(nx, ny, board.x_size);
		if (board.isOnBoard(nx, ny))
		{
			Piece pc = board.colors[nl];
			if (pc != C_EMPTY && getSide(pc) == pla)
				if (getHorizontal(pc) && isGoldMove(pc))return true;
		}
	}
	//check silver
	for (int i : silver)
	{
		int nx = x, ny = y, j = rotInd[i];
		if (pla == P_WHITE)nx += board.adj_ofx[i], ny += board.adj_ofy[i];
		else nx -= board.adj_ofx[i], ny -= board.adj_ofy[i];
		Loc nl = Location::getLoc(nx, ny, board.x_size);
		if (board.isOnBoard(nx, ny))
		{
			Piece pc = board.colors[nl];
			if (pc != C_EMPTY && getSide(pc) == pla)
				if (!getPromoted(pc) && !getHorizontal(pc) && getType(pc) == S_SILVER)return true;
		}
		nx = x; ny = y;
		if (pla == P_WHITE)nx += board.adj_ofx[j], ny += board.adj_ofy[j];
		else nx -= board.adj_ofx[j], ny -= board.adj_ofy[j];
		nl = Location::getLoc(nx, ny, board.x_size);
		if (board.isOnBoard(nx, ny))
		{
			Piece pc = board.colors[nl];
			if (pc != C_EMPTY && getSide(pc) == pla)
				if (!getPromoted(pc) && getHorizontal(pc) && getType(pc) == S_SILVER)return true;
		}
	}
	//check King, Promoted Bishop and Promoted Rook
	for (int i = 0; i < 8; ++i)
	{
		int nx = x, ny = y;
		nx += board.adj_ofx[i], ny += board.adj_ofy[i];
		Loc nl = Location::getLoc(nx, ny, board.x_size);
		if (board.isOnBoard(nx, ny))
		{
			Piece pc = board.colors[nl];
			Piece tp = getType(pc);
			if (pc != C_EMPTY && getSide(pc) == pla)
				if (tp == S_KING || (getPromoted(pc) && (tp == S_BISHOP || tp == S_ROOK)))return true;
		}
	}
	//check Knight
	int k_ofx[]{ -1,1,-2,-2 }, k_ofy[]{ -2,-2,-1,1 };
	for (int i = 0; i < 2; ++i)
	{
		int nx = x, ny = y;
		if (pla == P_WHITE)nx += k_ofx[i], ny += k_ofy[i];
		else nx -= k_ofx[i], ny -= k_ofy[i];
		Loc nl = Location::getLoc(nx, ny, board.x_size);
		if (board.isOnBoard(nx, ny))
		{
			Piece pc = board.colors[nl];
			Piece tp = getType(pc);
			if (pc != C_EMPTY && getSide(pc) == pla)
				if (!getPromoted(pc) && !getHorizontal(pc) && tp == S_KNIGHT)return true;
		}
	}
	for (int i = 2; i < 4; ++i)
	{
		int nx = x, ny = y;
		if (pla == P_WHITE)nx += k_ofx[i], ny += k_ofy[i];
		else nx -= k_ofx[i], ny -= k_ofy[i];
		Loc nl = Location::getLoc(nx, ny, board.x_size);
		if (board.isOnBoard(nx, ny))
		{
			Piece pc = board.colors[nl];
			Piece tp = getType(pc);
			if (pc != C_EMPTY && getSide(pc) == pla)
				if (!getPromoted(pc) && getHorizontal(pc) && tp == S_KNIGHT)return true;
		}
	}
	//check Pawn and Lance

	for (int i : pawn)
	{
		int nx = x, ny = y, j = rotInd[i];
		int ofx = board.adj_ofx[i], ofy = board.adj_ofy[i];
		if (pla != P_WHITE)ofx = -ofx, ofy = -ofy;
		nx += ofx; ny += ofy;
		Loc nl = Location::getLoc(nx, ny, board.x_size);
		if (board.isOnBoard(nx, ny))
		{
			Piece pc = board.colors[nl];
			Piece tp = getType(pc);
			if (pc != C_EMPTY && getSide(pc) == pla)
				if (!getHorizontal(pc) && !getPromoted(pc) && (tp == S_LANCE || tp == S_PAWN))return true;
		}
		if (board.colors[nl] == C_EMPTY)while (true)
		{
			nx += ofx; ny += ofy;
			Loc nl = Location::getLoc(nx, ny, board.x_size);
			if (!board.isOnBoard(nx, ny))break;
			Piece pc = board.colors[nl];
			if (pc == C_EMPTY)continue;
			Piece tp = getType(pc);
			if (getSide(pc) == pla)
				if (!getHorizontal(pc) && !getPromoted(pc) && tp == S_LANCE)return true;
			break;
		}

		nx = x; ny = y;
		ofx = board.adj_ofx[j], ofy = board.adj_ofy[j];
		if (pla != P_WHITE)ofx = -ofx, ofy = -ofy;
		nx += ofx; ny += ofy;
		nl = Location::getLoc(nx, ny, board.x_size);
		if (board.isOnBoard(nx, ny))
		{
			Piece pc = board.colors[nl];
			Piece tp = getType(pc);
			if (pc != C_EMPTY && getSide(pc) == pla)
				if (getHorizontal(pc) && !getPromoted(pc) && (tp == S_LANCE || tp == S_PAWN))return true;
		}
		if (board.colors[nl] == C_EMPTY)while (true)
		{
			nx += ofx; ny += ofy;
			Loc nl = Location::getLoc(nx, ny, board.x_size);
			if (!board.isOnBoard(nx, ny))break;
			Piece pc = board.colors[nl];
			if (pc == C_EMPTY)continue;
			Piece tp = getType(pc);
			if (getSide(pc) == pla)
				if (getHorizontal(pc) && !getPromoted(pc) && tp == S_LANCE)return true;
			break;
		}
	}
	//check Rook
	for (int i = 0; i < 4; ++i)
	{
		int nx = x, ny = y;
		int ofx = board.adj_ofx[i], ofy = board.adj_ofy[i];
		if (pla != P_WHITE)ofx = -ofx, ofy = -ofy;
		nx += ofx; ny += ofy;
		while (board.isOnBoard(nx, ny))
		{
			Loc nl = Location::getLoc(nx, ny, board.x_size);
			Piece pc = board.colors[nl];
			nx += ofx; ny += ofy;

			if (pc == C_EMPTY)continue;
			Piece tp = getType(pc);
			if (getSide(pc) == pla)
				if (tp == S_ROOK)return true;
			break;
		}
	}
	//check Bishop
	for (int i = 4; i < 8; ++i)
	{
		int nx = x, ny = y;
		int ofx = board.adj_ofx[i], ofy = board.adj_ofy[i];
		if (pla != P_WHITE)ofx = -ofx, ofy = -ofy;
		nx += ofx; ny += ofy;
		while (board.isOnBoard(nx, ny))
		{
			Loc nl = Location::getLoc(nx, ny, board.x_size);
			Piece pc = board.colors[nl];
			nx += ofx; ny += ofy;
			if (pc == C_EMPTY)continue;
			Piece tp = getType(pc);
			if (getSide(pc) == pla)
				if (tp == S_BISHOP)return true;
			break;
		}
	}
	return false;

	//auto slider = [&](int adj, function<bool(const Piece&)> checkFunc) {
	//  for(Loc tloc = loc + adj; board.isOnBoard(tloc); tloc += adj)
	//    if(board.colors[tloc] != C_EMPTY)
	//      return checkFunc(board.colors[tloc]);
	//  return false;
	//};
	//auto shifter = [&](int adj, function<bool(const Piece&)> checkFunc) {
	//  Loc tloc = loc + adj;
	//  if(!board.isOnBoard(tloc) || board.colors[tloc] == C_EMPTY)
	//    return false;
	//  return checkFunc(board.colors[tloc]);
	//};
	//function<bool(const Piece&)> cf[]{
	//  [&](const Piece& pc) {
	//    if(getSide(pc) != pla)
	//      return false;
	//    Piece tp = getType(pc);
	//    return tp == S_GOLD || tp == S_KING || tp == S_ROOK || getPromoted(pc);
	//  },
	//  [&](const Piece& pc) {
	//    if(getSide(pc) != pla)
	//      return false;
	//    Piece tp = getType(pc);
	//    return tp == S_GOLD || tp == S_KING || tp == S_ROOK ||
	//           (getHorizontal(pc) && (tp == S_SILVER || tp == S_LANCE || tp == S_PAWN)) || getPromoted(pc);
	//  },
	//  [&](const Piece& pc) {
	//    if(getSide(pc) != pla)
	//      return false;
	//    Piece tp = getType(pc);
	//    return tp == S_GOLD || tp == S_KING || tp == S_ROOK || getPromoted(pc);
	//  },
	//  [&](const Piece& pc) {
	//    if(getSide(pc) != pla)
	//      return false;
	//    Piece tp = getType(pc);
	//    return tp == S_GOLD || tp == S_KING || tp == S_ROOK ||
	//           (!getHorizontal(pc) && (tp == S_SILVER || tp == S_LANCE || tp == S_PAWN)) || getPromoted(pc);
	//  },
	//  [&](const Piece& pc) {
	//    if(getSide(pc) != pla)
	//      return false;
	//    Piece tp = getType(pc);
	//    if(getPromoted(pc) || tp == S_GOLD)
	//      return getHorizontal(pc) || tp == S_ROOK;
	//    return tp == S_BISHOP || tp == S_SILVER || tp == S_KING;
	//  },
	//  [&](const Piece& pc) {
	//    if(getSide(pc) != pla)
	//      return false;
	//    Piece tp = getType(pc);
	//    if(getPromoted(pc) || tp == S_GOLD)
	//      return tp == S_ROOK;
	//    return tp == S_BISHOP || tp == S_SILVER || tp == S_KING;
	//  },
	//  [&](const Piece& pc) {
	//    if(getSide(pc) != pla)
	//      return false;
	//    Piece tp = getType(pc);
	//    return getPromoted(pc) || tp == S_GOLD || tp == S_BISHOP || tp == S_SILVER || tp == S_KING;
	//  },
	//  [&](const Piece& pc) {
	//    if(getSide(pc) != pla)
	//      return false;
	//    Piece tp = getType(pc);
	//    if(getPromoted(pc) || tp == S_GOLD)
	//      return !getHorizontal(pc) || tp == S_ROOK;
	//    return tp == S_BISHOP || tp == S_SILVER || tp == S_KING;
	//  }};
	//for(int i = 0; i < 8; ++i) {
	//  int ind = pla == P_WHITE ? i : i ^ 3;
	//  if(shifter(board.adj_offsets[ind], cf[i]))
	//    return true;
	//}
	//function<bool(const Piece&)> cf2[]{[&](const Piece& pc) {
	//                                     if(getSide(pc) != pla)
	//                                       return false;
	//                                     Piece tp = getType(pc);
	//                                     return tp == S_ROOK;
	//                                   },
	//                                   [&](const Piece& pc) {
	//                                     if(getSide(pc) != pla)
	//                                       return false;
	//                                     Piece tp = getType(pc);
	//                                     return tp == S_ROOK || (getHorizontal(pc) && !getPromoted(pc) && tp == S_LANCE);
	//                                   },
	//                                   [&](const Piece& pc) {
	//                                     if(getSide(pc) != pla)
	//                                       return false;
	//                                     Piece tp = getType(pc);
	//                                     return tp == S_ROOK;
	//                                   },
	//                                   [&](const Piece& pc) {
	//                                     if(getSide(pc) != pla)
	//                                       return false;
	//                                     Piece tp = getType(pc);
	//                                     return tp == S_ROOK || (!getHorizontal(pc) && !getPromoted(pc) && tp == S_LANCE);
	//                                   },
	//                                   [&](const Piece& pc) {
	//                                     if(getSide(pc) != pla)
	//                                       return false;
	//                                     Piece tp = getType(pc);
	//                                     return tp == S_BISHOP;
	//                                   },
	//                                   [&](const Piece& pc) {
	//                                     if(getSide(pc) != pla)
	//                                       return false;
	//                                     Piece tp = getType(pc);
	//                                     return tp == S_BISHOP;
	//                                   },
	//                                   [&](const Piece& pc) {
	//                                     if(getSide(pc) != pla)
	//                                       return false;
	//                                     Piece tp = getType(pc);
	//                                     return tp == S_BISHOP;
	//                                   },
	//                                   [&](const Piece& pc) {
	//                                     if(getSide(pc) != pla)
	//                                       return false;
	//                                     Piece tp = getType(pc);
	//                                     return tp == S_BISHOP;
	//                                   }};
	//for(int i = 0; i < 8; ++i) {
	//  int ind = pla == P_WHITE ? i : i ^ 3;
	//  if(slider(board.adj_offsets[ind], cf2[i]))
	//    return true;
	//}
	//function<bool(const Piece&)> ncf[]{[&](const Piece& pc) {
	//                                     if(getSide(pc) != pla)
	//                                       return false;
	//                                     return !getHorizontal(pc) && !getPromoted(pc) && getType(pc) == S_KNIGHT;
	//                                   },
	//                                   [&](const Piece& pc) {
	//                                     if(getSide(pc) != pla)
	//                                       return false;
	//                                     return getHorizontal(pc) && !getPromoted(pc) && getType(pc) == S_KNIGHT;
	//                                   }};
	//int ss[]{board.adj_offsets[3] + board.adj_offsets[6],
	//         board.adj_offsets[3] + board.adj_offsets[7],
	//         board.adj_offsets[1] + board.adj_offsets[4],
	//         board.adj_offsets[1] + board.adj_offsets[6]};
	//int inds[]{0, 0, 1, 1};
	//for(int i = 0; i < 4; ++i) {
	//  int adj = pla == P_WHITE ? ss[i] : -ss[i];
	//  if(shifter(adj, ncf[i >= 2]))
	//    return true;
	//}
	//return false;
}

//bool GameLogic::checkLegal(const Board& board, Loc loc, Piece piece, Loc locd, Player pla) {
//  int x = Location::getX(loc, board.x_size);
//  int y = Location::getY(loc, board.x_size);
//  if(!board.isOnBoard(loc) && loc != Board::PASS_LOC)
//    return false;
//  if(!board.isOnBoard(locd))
//    return false;
//  int disx = Location::getX(locd, board.x_size), disy = Location::getY(locd, board.x_size);
//  auto shifter = [&](int adj) {
//    Loc tloc = loc + adj;
//    if(tloc != locd)
//      return false;
//    // Board::printlocation(tloc, board, "tloc:");
//    if(board.isOnBoard(tloc) && getSide(board.colors[tloc]) != pla)
//      return true;
//    return false;
//  };
//  auto slider = [&](int adj) {
//    int dx, dy, cx = 0, cy = 0;
//    getdXY(adj, board.x_size, dx, dy);
//    // cout << adj << ' ' << dx << ' ' << dy << endl;
//    int diffx = abs(disx - x), diffy = abs(disy - y);
//    if(dx != 0) {
//      if((dx < 0 && disx >= x) || (dx > 0 && disx <= x))
//        return false;
//      int absdx = abs(dx);
//      if(diffx % absdx)
//        return false;
//      cx = diffx / absdx;
//    }
//    if(dy != 0) {
//      if((dy < 0 && disy >= y) || (dy > 0 && disy <= y))
//        return false;
//      int absdx = abs(dy);
//      if(diffy % absdx)
//        return false;
//      cy = diffy / absdx;
//    }
//    if((dx == 0 && diffx == 0) || (dy == 0 && diffy == 0) || cx == cy) {
//      for(Loc l = loc + adj; l != locd; l += adj) {
//        assert(board.isOnBoard(l));
//        if(board.colors[l] != C_EMPTY)
//          return false;
//      }
//      return true;
//    }
//    return false;
//  };
//  auto goldmove = [&]() {
//    int s[]{0, 1, 2, 3, 6, 7};
//    for(int i: s) {
//      int adj = board.adj_offsets[i];
//      if(getHorizontal(piece))
//        adj = getRotatedOffset(adj, board.x_size);
//      if(getSide(piece) == P_WHITE)
//        adj = getRotatedOffset(adj, board.x_size, 2);
//      if(shifter(adj))
//        return true;
//    }
//    return false;
//  };
//  if(loc != Board::PASS_LOC) {
//    bool isPromoting = board.changeState;
//    if(
//      isPromoting && !inPromotionArea(board, piece, loc, board.x_size, board.y_size) &&
//      !inPromotionArea(board, piece, locd, board.x_size, board.y_size))
//      return false;
//    if(getSide(board.colors[Location::getLoc(disx, disy, board.x_size)]) == pla)
//      return false;
//    vector<int> s;
//    switch(getType(piece)) {
//      case S_KING:
//        for(auto i: board.adj_offsets)
//          if(shifter(i))
//            return true;
//        break;
//      case S_GOLD:
//        if(goldmove())
//          return true;
//        break;
//      case S_SILVER:
//        if(getPromoted(piece)) {
//          if(goldmove())
//            return true;
//        } else {
//          s = vector<int>({3, 4, 5, 6, 7});
//          for(int i: s) {
//            int adj = board.adj_offsets[i];
//            if(getHorizontal(piece))
//              adj = getRotatedOffset(adj, board.x_size);
//            if(getSide(piece) == P_WHITE)
//              adj = getRotatedOffset(adj, board.x_size, 2);
//            if(shifter(adj))
//              return true;
//          }
//        }
//        break;
//      case S_PAWN:
//        if(getPromoted(piece)) {
//          if(goldmove())
//            return true;
//        } else {
//          s = vector<int>({3});
//          for(int i: s) {
//            int adj = board.adj_offsets[i];
//            if(getHorizontal(piece))
//              adj = getRotatedOffset(adj, board.x_size);
//            if(getSide(piece) == P_WHITE)
//              adj = getRotatedOffset(adj, board.x_size, 2);
//            if(!isPromoting && !board.isOnBoard(locd + adj))
//              return false;
//            if(shifter(adj))
//              return true;
//          }
//        }
//        break;
//      case S_LANCE:
//        if(getPromoted(piece)) {
//          if(goldmove())
//            return true;
//        } else {
//          s = vector<int>({3});
//          for(int i: s) {
//            int adj = board.adj_offsets[i];
//            if(getHorizontal(piece))
//              adj = getRotatedOffset(adj, board.x_size);
//            if(getSide(piece) == P_WHITE)
//              adj = getRotatedOffset(adj, board.x_size, 2);
//            if(!isPromoting && !board.isOnBoard(locd + adj))
//              return false;
//            if(slider(adj))
//              return true;
//          }
//        }
//        break;
//      case S_BISHOP:
//        s = vector<int>({4, 5, 6, 7});
//        for(int i: s) {
//          int adj = board.adj_offsets[i];
//          if(slider(adj))
//            return true;
//        }
//        if(getPromoted(piece)) {
//          int s2[]{0, 1, 2, 3};
//          for(int i: s2) {
//            int adj = board.adj_offsets[i];
//            if(shifter(adj))
//              return true;
//          }
//        }
//        break;
//      case S_ROOK:
//        s = vector<int>({0, 1, 2, 3});
//        for(int i: s) {
//          int adj = board.adj_offsets[i];
//          if(slider(adj))
//            return true;
//        }
//        if(getPromoted(piece)) {
//          int s2[]{4, 5, 6, 7};
//          for(int i: s2) {
//            int adj = board.adj_offsets[i];
//            if(shifter(adj))
//              return true;
//          }
//        }
//        break;
//      case S_KNIGHT:
//        if(getPromoted(piece)) {
//          if(goldmove())
//            return true;
//        } else {
//          s = vector<int>({board.adj_offsets[3] + board.adj_offsets[6], board.adj_offsets[3] + board.adj_offsets[7]});
//          for(int i: s) {
//            int adj = i;
//            if(getHorizontal(piece))
//              adj = getRotatedOffset(adj, board.x_size);
//            if(getSide(piece) == P_WHITE)
//              adj = getRotatedOffset(adj, board.x_size, 2);
//            if((isPromoting || !inKnightMustPromoteArea(board, piece, locd)) && shifter(adj))
//              return true;
//          }
//        }
//        break;
//      default:
//        return false;
//    }
//    return false;
//  } else {
//    Piece tp = getType(piece);
//    int nx = disx, ny = disy;
//    int start = board.x_size;
//    if(board.colors[locd] != C_EMPTY || getPromoted(piece) || getHorizontal(piece) || tp == S_KING)
//      return false;
//    if(tp == S_LANCE || tp == S_PAWN)
//      --start;
//    else if(tp == S_KNIGHT)
//      start -= 2;
//    if(!board.changeState) {
//      if(pla == P_WHITE)
//        ny = board.y_size - 1 - disy;
//      if(nx < 0 || nx >= board.x_size || ny < 0 || ny >= start)
//        return false;
//      if((tp == S_PAWN || tp == S_LANCE) && ny == start - 1 && (nx == 0 || nx == board.x_size - 1))
//        return false;
//      if(tp == S_PAWN) {
//        for(int yy = 0; yy < board.y_size; ++yy) {
//          int dloc = Location::getLoc(disx, yy, board.x_size);
//          Piece pc = board.colors[dloc];
//          if(getType(pc) == S_PAWN && getSide(pc) == pla && !getHorizontal(pc) && !getPromoted(pc)) {
//            return false;
//          }
//        }
//        int aloc = Location::getLoc(disx, disy, board.x_size) + board.adj_offsets[pla == P_WHITE ? 0 : 3];
//        // cout << Location::getX(aloc, board.x_size) << ' ' << Location::getY(aloc, board.y_size);
//        Piece mk = board.colors[aloc];
//        if(getType(mk) == S_KING && getSide(mk) != pla) {
//          bool flag = false;
//          for(int i = 0; i < 8; ++i) {
//            Loc naloc = aloc + board.adj_offsets[i];
//            if(
//              board.isOnBoard(naloc) && getSide(board.colors[naloc]) != getOpp(pla) &&
//              !GameLogic::locAttacked(board, naloc, pla)) {
//              // cout << GameLogic::locAttacked(board, naloc, pla) << endl;
//              flag = true;
//              break;
//            }
//          }
//          if(!flag) {
//            vector<Loc> vec;
//            if(GameLogic::getAttackingPiece(board, getOpp(pla), locd, vec)) {
//              Board cboard(board);
//              for(const Loc& lc: vec) {
//                if(flag)
//                  break;
//                if(getType(board.colors[lc]) == S_KING && getSide(board.colors[lc]) != pla)
//                  continue;
//                cboard.colors[locd] = cboard.colors[lc];
//                cboard.colors[lc] = C_EMPTY;
//                flag |= !GameLogic::locAttacked(cboard, aloc, pla);
//                cboard.colors[lc] = cboard.colors[locd];
//                cboard.colors[locd] = C_EMPTY;
//              }
//            }
//          }
//          if(!flag)
//            return false;
//        }
//      }
//    } else {
//      if(pla == P_BLACK)
//        nx = board.x_size - 1 - disx;
//      if(nx < 0 || nx >= start || ny < 0 || ny >= board.y_size)
//        return false;
//      if((tp == S_PAWN || tp == S_LANCE) && nx == start - 1 && (ny == 0 || ny == board.x_size - 1))
//        return false;
//      if(tp == S_PAWN) {
//        for(int xx = 0; xx < board.x_size; ++xx) {
//          int dloc = Location::getLoc(xx, disy, board.x_size);
//          Piece pc = board.colors[dloc];
//          if(getType(pc) == S_PAWN && getSide(pc) == pla && getHorizontal(pc) && !getPromoted(pc)) {
//            return false;
//          }
//        }
//      }
//      int aloc = Location::getLoc(disx, disy, board.x_size) + board.adj_offsets[pla == P_WHITE ? 1 : 2];
//      Piece mk = board.colors[aloc];
//      if(getType(mk) == S_KING && getSide(mk) != pla) {
//        bool flag = false;
//        for(int i = 0; i < 8; ++i) {
//          Loc naloc = aloc + board.adj_offsets[i];
//          if(
//            board.isOnBoard(naloc) && getSide(board.colors[naloc]) != getOpp(pla) &&
//            !GameLogic::locAttacked(board, naloc, pla)) {
//            flag = true;
//            break;
//          }
//        }
//        if(!flag) {
//          vector<Loc> vec;
//          if(GameLogic::getAttackingPiece(board, getOpp(pla), locd, vec)) {
//            Board cboard(board);
//            for(const Loc& lc: vec) {
//              if(flag)
//                break;
//              if(getType(board.colors[lc]) == S_KING && getSide(board.colors[lc]) != pla)
//                continue;
//              cboard.colors[locd] = cboard.colors[lc];
//              cboard.colors[lc] = C_EMPTY;
//              flag |= !GameLogic::locAttacked(cboard, aloc, pla);
//              cboard.colors[lc] = cboard.colors[locd];
//              cboard.colors[locd] = C_EMPTY;
//            }
//          }
//        }
//        if(!flag)
//          return false;
//      }
//    }
//    return true;
//  }
//  return false;
//}

// bool GameLogic::getMove(const Board& board, vector<Loc>& res, Loc loc, Piece piece) {
//  int x = Location::getX(loc, board.x_size);
//  int y = Location::getY(loc, board.x_size);
//  res.clear();
//  Player pla = getSide(piece);
//  if(loc == Board::PASS_LOC)
//    return false;
//  if(!board.isOnBoard(loc) && y != board.y_size * 2)
//    return false;
//  if(y != board.y_size * 2) {
//    auto slider = [&](int adj) {
//      for(Loc tloc = loc + adj; board.isOnBoard(tloc) && getSide(board.colors[tloc]) != pla; tloc += adj) {
//        res.push_back(tloc);
//        if(board.colors[tloc] != C_EMPTY)
//          break;
//      }
//    };
//    auto shifter = [&](int adj) {
//      Loc tloc = loc + adj;
//      if(board.isOnBoard(tloc) && getSide(board.colors[tloc]) != pla)
//        res.push_back(tloc);
//    };
//    auto goldmove = [&]() {
//      int s[]{0, 1, 2, 3, 6, 7};
//      for(int i: s) {
//        int adj = s[i];
//        if(getHorizontal(piece))
//          adj = getRotatedOffset(adj, board.x_size);
//        if(getSide(piece) == P_WHITE)
//          adj = getRotatedOffset(adj, board.x_size, 2);
//        shifter(adj);
//      }
//    };
//    switch(getType(piece)) {
//      case S_KING:
//        for(auto i: board.adj_offsets)
//          shifter(i);
//        break;
//      case S_GOLD:
//        goldmove();
//        break;
//      case S_SILVER:
//        if(getPromoted(piece))
//          goldmove();
//        else {
//          int s[]{3, 4, 5, 6, 7};
//          for(int i: s) {
//            int adj = board.adj_offsets[i];
//            if(getHorizontal(piece))
//              adj = getRotatedOffset(adj, board.x_size);
//            if(getSide(piece) == P_WHITE)
//              adj = getRotatedOffset(adj, board.x_size, 2);
//            shifter(adj);
//          }
//        }
//        break;
//      case S_PAWN:
//        if(getPromoted(piece))
//          goldmove();
//        else {
//          int s[]{3};
//          for(int i: s) {
//            int adj = board.adj_offsets[i];
//            if(getHorizontal(piece))
//              adj = getRotatedOffset(adj, board.x_size);
//            if(getSide(piece) == P_WHITE)
//              adj = getRotatedOffset(adj, board.x_size, 2);
//            shifter(adj);
//          }
//        }
//        break;
//      case S_LANCE:
//        if(getPromoted(piece))
//          goldmove();
//        else {
//          int s[]{3};
//          for(int i: s) {
//            int adj = board.adj_offsets[i];
//            if(getHorizontal(piece))
//              adj = getRotatedOffset(adj, board.x_size);
//            if(getSide(piece) == P_WHITE)
//              adj = getRotatedOffset(adj, board.x_size, 2);
//            slider(adj);
//          }
//        }
//        break;
//      case S_BISHOP:
//        int s[]{4, 5, 6, 7};
//        for(int i: s) {
//          int adj = board.adj_offsets[i];
//          slider(adj);
//        }
//        if(getPromoted(piece)) {
//          int s2[]{0, 1, 2, 3};
//          for(int i: s2) {
//            int adj = board.adj_offsets[i];
//            shifter(adj);
//          }
//        }
//        break;
//      case S_ROOK:
//        int s[]{0, 1, 2, 3};
//        for(int i: s) {
//          int adj = board.adj_offsets[i];
//          slider(adj);
//        }
//        if(getPromoted(piece)) {
//          int s2[]{4, 5, 6, 7};
//          for(int i: s2) {
//            int adj = board.adj_offsets[i];
//            shifter(adj);
//          }
//        }
//        break;
//      case S_KNIGHT:
//        if(getPromoted(piece))
//          goldmove();
//        else {
//          int s[]{3};
//          for(int i: s) {
//            int adj = board.adj_offsets[i];
//            if(getHorizontal(piece))
//              adj = getRotatedOffset(adj, board.x_size);
//            if(getSide(piece) == P_WHITE)
//              adj = getRotatedOffset(adj, board.x_size, 2);
//            shifter(adj);
//          }
//          vector<Loc> tmp(res);
//          res.clear();
//          for(Loc ll: tmp) {
//            int s2[]{6, 7};
//            for(int i: s2) {
//              int adj = board.adj_offsets[i];
//              if(getHorizontal(piece))
//                adj = getRotatedOffset(adj, board.x_size);
//              if(getSide(piece) == P_WHITE)
//                adj = getRotatedOffset(adj, board.x_size, 2);
//              Loc tmploc = loc;
//              loc = ll;
//              shifter(adj);
//              loc = tmploc;
//            }
//          }
//        }
//        break;
//      default:
//        return false;
//    }
//  } else {
//    int start = board.x_size;
//    if(getPromoted(piece) || getHorizontal(piece) || getType(piece) == S_KING)
//      return false;
//    if(getType(piece) == S_LANCE || getType(piece) == S_PAWN)
//      --start;
//    else if(getType(piece) == S_KNIGHT)
//      start -= 2;
//    for(int nx = 0; nx < board.x_size; ++nx) {
//      bool flag = false;
//      for(int yy = 0; yy < board.y_size; ++yy) {
//        int ny = yy;
//        if(pla == P_WHITE)
//          ny = board.y_size - 1 - yy;
//        int dloc = Location::getLoc(nx, ny, board.x_size);
//        Piece pc = board.colors[dloc];
//        if(getType(pc) == S_PAWN && getSide(pc) == pla && !getHorizontal(pc) && !getPromoted(pc)) {
//          flag = true;
//          break;
//        }
//      }
//      if(flag)
//        continue;
//      for(int yy = 0; yy < start; ++yy) {
//        int ny = yy;
//        if(pla == P_WHITE)
//          ny = board.y_size - 1 - yy;
//        int dloc = Location::getLoc(nx, ny, board.x_size);
//        if(!getType(board.colors[dloc]))
//          res.push_back(dloc);
//      }
//    }
//    for(int ny = 0; ny < board.y_size; ++ny) {
//      bool flag = false;
//      for(int xx = 0; xx < board.x_size; ++xx) {
//        int nx = xx;
//        if(pla == P_BLACK)
//          nx = board.x_size - 1 - xx;
//        int dloc = Location::getLoc(nx, ny, board.x_size);
//        Piece pc = board.colors[dloc];
//        if(getType(pc) == S_PAWN && getSide(pc) == pla && getHorizontal(pc) && !getPromoted(pc)) {
//          flag = true;
//          break;
//        }
//      }
//      if(flag)
//        continue;
//      for(int xx = 0; xx < start; ++xx) {
//        int nx = xx;
//        if(pla == P_BLACK)
//          nx = board.x_size - 1 - xx;
//        int dloc = Location::shiftBoard(nx, ny, board.x_size, board.y_size);
//        if(!getType(board.colors[dloc]))
//          res.push_back(dloc);
//      }
//    }
//  }
//}



Piece GameLogic::checkWinnerAfterPlayed(const Board& board, Player pla, Loc loc) {
	if (loc == Board::PASS_LOC)
		return getOpp(pla);  // pass is not allowed
	if (locAttacked(board, board.kings[pla], getOpp(pla)))
		return getOpp(pla);
	return board.winner;
}
