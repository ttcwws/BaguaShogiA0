#pragma once
#include "board.h"
#include "gamelogic.h"
/*
 * board.cpp
 * Originally from an unreleased project back in 2010, modified since.
 * Authors: brettharrison (original), David Wu (original and later modificationss).
 */

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>
#include <exception>
#include <unordered_set>
#define ASSERT_UNREACHABLE assert(false);



	using namespace std;

//STATIC VARS-----------------------------------------------------------------------------
bool Board::IS_ZOBRIST_INITALIZED = false;
Hash128 Board::ZOBRIST_SIZE_X_HASH[MAX_LEN + 1];
Hash128 Board::ZOBRIST_SIZE_Y_HASH[MAX_LEN + 1];
Hash128 Board::ZOBRIST_BOARD_HASH[MAX_ARR_SIZE][NUM_BOARD_COLORS];
Hash128 Board::ZOBRIST_STAGENUM_HASH[STAGE_NUM_EACH_PLA];
Hash128 Board::ZOBRIST_STAGELOC_HASH[MAX_ARR_SIZE * 2][STAGE_NUM_EACH_PLA];
Hash128 Board::ZOBRIST_NEXTPLA_HASH[4];
Hash128 Board::ZOBRIST_MOVENUM_HASH[MAX_MOVE_NUM];
Hash128 Board::ZOBRIST_KOMA_HASH[4][MAX_KOMA_SIZE][MAX_KOMA_NUM];
Hash128 Board::ZOBRIST_REPETITION_HASH[MAX_REPETITION_NUM + 1];
Hash128 Board::ZOBRIST_REPLOSS_HASH[2];
Hash128 Board::ZOBRIST_PLAYER_HASH[4];
//Loc Board::KING_INIT_LOCATION[4];
short Board::adj_ofx[8];
short Board::adj_ofy[8];
const Hash128 Board::ZOBRIST_GAME_IS_OVER = Hash128(0xb6f9e465597a77eeULL, 0xf1d583d960a4ce7fULL);

std::string to_string(Move mv)
{
	using namespace Location;
	int x1 = getX(mv.x, 9), y1 = getY(mv.x, 9);
	std::string s;
	if (mv.z)
	{
		char c = '?';
		switch (mv.z)
		{
		case S_GOLD:
			c = 'G'; break;
		case S_SILVER:
			c = 'S'; break;
		case S_LANCE:
			c = 'L'; break;
		case S_BISHOP:
			c = 'B'; break;
		case S_ROOK:
			c = 'R'; break;
		case S_PAWN:
			c = 'P'; break;
		case S_KNIGHT:
			c = 'N'; break;
		default:break;
		}
		s += c;
		s += x1 + 'A';
		s += '9' - y1;
		s += mv.change ? '=' : '~';
		return s;
	}
	int x2 = getX(mv.y, 9), y2 = getY(mv.y, 9);
	s += x1 + 'A';
	s += '9' - y1;
	s += x2 + 'A';
	s += '9' - y2;
	if (mv.change)s += '+';
	return s;
}
namespace Location
{
	std::vector<Move> policy;
	int movePolicy[(BOARD_AREA + S_NUM_PIECES - 1) * BOARD_AREA * 2];
	std::vector<std::tuple<int, int, int>> validPolicy;
}
//LOCATION--------------------------------------------------------------------------------
Loc Location::getLoc(int x, int y, int x_size) {
	return x + y * x_size;
}
int Location::getX(Loc loc, int x_size)
{
	return loc % x_size;
}
int Location::getY(Loc loc, int x_size)
{
	return loc / x_size;
}
void Location::getAdjacentOffsets(short adj_offsets[8], int x_size) {
	adj_offsets[0] = -x_size;
	adj_offsets[1] = -1;
	adj_offsets[2] = 1;
	adj_offsets[3] = x_size;
	adj_offsets[4] = -x_size - 1;
	adj_offsets[5] = -x_size + 1;
	adj_offsets[6] = x_size - 1;
	adj_offsets[7] = x_size + 1;

}
Loc Location::shiftBoard(int x, int y, int x_size, int y_size) {
	return y < y_size ? getLoc(x, y + y_size, x_size) : getLoc(x, y - y_size, x_size);
}
void Board::printlocation(Loc loc, Board& board, string prompt) {
	int x = Location::getX(loc, board.x_size);
	int y = Location::getY(loc, board.x_size);
	//cout << prompt << x << ' ' << y << ' ' << endl;
}
bool Location::isAdjacent(Loc loc0, Loc loc1, int x_size)
{
	return loc0 == loc1 - (x_size + 1) || loc0 == loc1 - 1 || loc0 == loc1 + 1 || loc0 == loc1 + (x_size + 1);
}
Loc Location::RotLoc(Loc loc, int x_size)
{
	int x = getX(loc, x_size), y = getY(loc, x_size);
	return getLoc(x_size - 1 - x, x_size - 1 - y, x_size);
}

Move::operator unsigned()const
{
	static constexpr int shift = (BOARD_AREA + S_NUM_PIECES - 1) * BOARD_AREA;
	if (!z)return x * BOARD_AREA + y + change * shift;
	return (BOARD_AREA + z - 2) * BOARD_AREA + x + change * shift;
}
bool operator==(const Move& a, const Move& b)
{
	if (a.z != b.z)return false;
	if (a.z)return a.x == b.x && a.change == b.change;
	return a.x == b.x && a.y == b.y && a.change == b.change;
}
Move& Move::rot()
{
	x = Location::RotLoc(x, 9);
	y = Location::RotLoc(y, 9);
	return *this;
}
Move Move::rot_()
{
	Move ret(*this);
	ret.x = Location::RotLoc(x, 9);
	ret.y = Location::RotLoc(y, 9);
	return ret;
}
void Location::RotLocs(std::vector<int>& xs, std::vector<int>& ys)
{
	int len = xs.size();
	assert(xs.size() == ys.size());
	for (int i = 0; i < len; ++i)
	{
		int x = xs[i], y = ys[i];
		xs[i] = y;
		ys[i] = -x;
	}
}
bool Location::isOnBoard(const int& x, const int& y)
{
	static constexpr int len = COMPILE_MAX_BOARD_LEN;
	if (!x && y == len - 1 || x == len - 1 && !y)return false;
	return x >= 0 && x < len && y >= 0 && y < len;
}
/* 0~63 8 adj * 8
*  64~67 4 knight
*  68~74 7 drops
*  75~149 above + change
*/
void Location::policyInit()
{
	policy.clear();
	validPolicy.clear();
	policy.reserve(4790);
	validPolicy.reserve(4790);
	std::fill(movePolicy, end(movePolicy), -1);
	static constexpr int len = COMPILE_MAX_BOARD_LEN;
	for (int i = 0; i < len; ++i)
		for (int j = 0; j < len; ++j)
		{
			if (!i && j == len - 1 || i == len - 1 && !j)
				continue;
			Loc from = getLoc(i, j, len);
			//queen move
			constexpr char ofx[]{ 1,1,1,-1,-1,-1,0,0 }, ofy[]{ 1, 0, -1, 1, 0, -1, 1, -1 };
			for (int k = 0; k < 8; ++k)
			{
				for (int c = 0, nx = i + ofx[k], ny = j + ofy[k]; nx >= 0 && ny >= 0 && nx < len && ny < len; ++c, nx += ofx[k], ny += ofy[k])
				{
					if (!nx && ny == len - 1 || nx == len - 1 && !ny)
						break;
					Loc to = getLoc(nx, ny, len);
					Move mv(from, to);
					movePolicy[mv] = policy.size();
					// validPolicy stores (channel, source_x, source_y) in the transposed x-major grid layout.
					validPolicy.push_back({ k * 8 + c,i,j });
					policy.push_back(mv);
					if (i < 3 || j < 3 || nx < 3 || ny < 3)
					{
						Move pmv(from, to, 0, true);
						movePolicy[pmv] = policy.size();
						validPolicy.push_back({ k * 8 + c + 75,i,j });
						policy.push_back(pmv);
					}

				}
			}
			//knight move
			constexpr char kofx[]{ 1,-1,-2,-2 }, kofy[]{ -2,-2,1,-1 };
			for (int k = 0; k < 4; ++k)
			{
				int nx = i + kofx[k], ny = j + kofy[k];
				if (!nx && ny == len - 1 || nx == len - 1 && !ny)
					continue;
				if (nx >= 0 && ny >= 0 && nx < len && ny < len)
				{
					Loc to = getLoc(nx, ny, len);
					Move mv(from, to);
					if ((k < 2 && ny >= 2) || (k >= 2 && nx >= 2))
					{
						movePolicy[mv] = policy.size();
						validPolicy.push_back({ 64 + k,i,j });
						policy.push_back(mv);
					}
					if (i < 3 || j < 3 || nx < 3 || ny < 3)
					{
						Move pmv(from, to, 0, true);
						movePolicy[pmv] = policy.size();
						validPolicy.push_back({ 64 + k + 75,i,j });
						policy.push_back(pmv);
					}
				}
			}
			//dropping vertical
			for (int p = S_GOLD; p <= S_PAWN; ++p)
			{
				Move dp(from, 0, p);
				if ((p == S_KNIGHT && j < 2) ||
					((p == S_PAWN || p == S_LANCE) && !isOnBoard(i, j - 1)))continue;
				movePolicy[dp] = policy.size();
				validPolicy.push_back({ 68 + p - S_GOLD,i,j });
				policy.push_back(dp);
			}
			//dropping horizontal
			for (int p = S_GOLD; p <= S_PAWN; ++p)
			{
				Move dp(from, 0, p, true);
				if ((p == S_KNIGHT && i < 2) ||
					((p == S_PAWN || p == S_LANCE) && !isOnBoard(i - 1, j)))continue;
				movePolicy[dp] = policy.size();
				validPolicy.push_back({ 68 + p - S_GOLD + 75,i,j });
				policy.push_back(dp);
			}
		}
}
size_t Location::toPolicy(const Move& move)
{
	return movePolicy[move];
}
Loc Location::getCenterLoc(int x_size, int y_size) {
	if (x_size % 2 == 0 || y_size % 2 == 0)
		return Board::NULL_LOC;
	return getLoc(x_size / 2, y_size / 2, x_size);
}

Loc Location::getCenterLoc(const Board& b) {
	return getCenterLoc(b.x_size, b.y_size);
}

bool Location::isCentral(Loc loc, int x_size, int y_size) {
	int x = getX(loc, x_size);
	int y = getY(loc, x_size);
	return x >= (x_size - 1) / 2 && x <= x_size / 2 && y >= (y_size - 1) / 2 && y <= y_size / 2;
}

bool Location::isNearCentral(Loc loc, int x_size, int y_size) {
	int x = getX(loc, x_size);
	int y = getY(loc, x_size);
	return x >= (x_size - 1) / 2 - 1 && x <= x_size / 2 + 1 && y >= (y_size - 1) / 2 - 1 && y <= y_size / 2 + 1;
}


#define FOREACHADJ(BLOCK) {int ADJOFFSET = -(x_size+1); {BLOCK}; ADJOFFSET = -1; {BLOCK}; ADJOFFSET = 1; {BLOCK}; ADJOFFSET = x_size+1; {BLOCK}};
#define ADJ0 (-(x_size+1))
#define ADJ1 (-1)
#define ADJ2 (1)
#define ADJ3 (x_size+1)

//CONSTRUCTORS AND INITIALIZATION----------------------------------------------------------

Board::Board()
{
	init(DEFAULT_LEN, DEFAULT_LEN);
}

Board::Board(int x, int y)
{
	init(x, y);
}


Board::Board(const Board& other)
{
	//for (auto& _ : checkRecord)
	//	_.clear(), _.push_back(0);
	x_size = other.x_size;
	y_size = other.y_size;

	memcpy(colors, other.colors, sizeof colors);
	memcpy(kings, other.kings, sizeof kings);
	memcpy(KING_INIT_LOCATION, other.KING_INIT_LOCATION, sizeof KING_INIT_LOCATION);
	movenum = other.movenum;
	raw_pos_hash = other.raw_pos_hash;
	memcpy(adj_offsets, other.adj_offsets, sizeof adj_offsets);
	memcpy(koma, other.koma, sizeof koma);
	dummy = other.dummy;
	//memcpy(KING_INIT_LOCATION, other.KING_INIT_LOCATION, sizeof KING_INIT_LOCATION);
	for (int i = 0; i < 2; ++i)
	{
		repetition[i] = other.repetition[i];
		first[i] = other.first[i];
	}

	for (int i = P_BLACK; i <= P_WHITE; ++i)
		checkRecord[i] = other.checkRecord[i], kings[i] = other.kings[i];
	winner = other.winner;
	attacked = other.attacked;
	nextPla = other.nextPla;

	//++repetition[0][raw_pos_hash.hash0];
	//++repetition[1][raw_pos_hash.hash1];
	//checkRecord[nextPla].push_back(GameLogic::locAttacked(*this, kings[getOpp(nextPla)], nextPla));
	//int repind = repetition[0][raw_pos_hash.hash0] < repetition[1][raw_pos_hash.hash1] ? 0 : 1;
	//auto rephash = repind ? raw_pos_hash.hash1 : raw_pos_hash.hash0;
	//first[repind][rephash] = checkRecord[nextPla].size() - 1;

}
Board::Board(const Board& other, bool temp)
{
	x_size = other.x_size;
	y_size = other.y_size;
	movenum = other.movenum;
	memcpy(colors, other.colors, sizeof colors);
	memcpy(kings, other.kings, sizeof kings);
	memcpy(KING_INIT_LOCATION, other.KING_INIT_LOCATION, sizeof KING_INIT_LOCATION);

	memcpy(adj_offsets, other.adj_offsets, sizeof adj_offsets);
	memcpy(koma, other.koma, sizeof koma);
	dummy = temp;
	attacked = other.attacked;
	winner = other.winner;
	nextPla = other.nextPla;
}

void Board::init(int xS, int yS) {
	assert(IS_ZOBRIST_INITALIZED);
	if (xS < 0 || yS < 0 || xS > MAX_LEN || yS > MAX_LEN)
		throw "Board::init - invalid board size";

	x_size = xS;
	y_size = yS;
	for (auto& _ : checkRecord)
		_.clear(), _.push_back(0);
	for (int i = 0; i < MAX_ARR_SIZE; i++)
		colors[i] = C_WALL;
	movenum = 0;
	winner = C_WALL;
	dummy = false;
	attacked = false;
	for (int y = 0; y < y_size; y++)
		for (int x = 0; x < x_size; x++) {
			Loc loc = Location::getLoc(x, y, x_size);
			colors[loc] = C_EMPTY;
			// empty_list.add(loc);
		}
	memset(koma, 0, sizeof koma);
	memset(KING_INIT_LOCATION, 0, sizeof KING_INIT_LOCATION);
	memset(kings, 0, sizeof kings);
	colors[Location::getLoc(0, y_size - 1, x_size)] = C_WALL;
	colors[Location::getLoc(x_size - 1, 0, x_size)] = C_WALL;
	nextPla = P_BLACK;
	raw_pos_hash = ZOBRIST_SIZE_X_HASH[x_size] ^ ZOBRIST_SIZE_Y_HASH[y_size] ^ ZOBRIST_NEXTPLA_HASH[nextPla];

	Location::getAdjacentOffsets(adj_offsets, x_size);
	adj_ofx[0] = 0; adj_ofy[0] = -1;
	adj_ofx[1] = -1; adj_ofy[1] = 0;
	adj_ofx[2] = 1; adj_ofy[2] = 0;
	adj_ofx[3] = 0; adj_ofy[3] = 1;
	adj_ofx[4] = -1; adj_ofy[4] = -1;
	adj_ofx[5] = 1; adj_ofy[5] = -1;
	adj_ofx[6] = -1; adj_ofy[6] = 1;
	adj_ofx[7] = 1; adj_ofy[7] = 1;


	setStone(Location::getLoc(8, 8, x_size), S_KING);
	setStone(Location::getLoc(7, 8, x_size), S_PAWN);
	setStone(Location::getLoc(6, 8, x_size), S_BISHOP);
	setStone(Location::getLoc(5, 8, x_size), S_GOLD);
	setStone(Location::getLoc(4, 8, x_size), S_SILVER);
	setStone(Location::getLoc(3, 8, x_size), S_KNIGHT);
	setStone(Location::getLoc(2, 8, x_size), S_LANCE);
	setStone(Location::getLoc(1, 8, x_size), S_PAWN);

	for (int i = 2; i <= 6; ++i)
		setStone(Location::getLoc(i, 7, x_size), S_PAWN);
	setStone(Location::getLoc(8, 7, x_size), S_PAWN ^ S_HORIZONTAL);
	setStone(Location::getLoc(8, 6, x_size), S_ROOK ^ S_HORIZONTAL);
	setStone(Location::getLoc(8, 5, x_size), S_GOLD ^ S_HORIZONTAL);
	setStone(Location::getLoc(8, 4, x_size), S_SILVER ^ S_HORIZONTAL);
	setStone(Location::getLoc(8, 3, x_size), S_KNIGHT ^ S_HORIZONTAL);
	setStone(Location::getLoc(8, 2, x_size), S_LANCE ^ S_HORIZONTAL);
	setStone(Location::getLoc(8, 1, x_size), S_PAWN ^ S_HORIZONTAL);
	for (int i = 2; i <= 6; ++i)
		setStone(Location::getLoc(7, i, x_size), S_PAWN ^ S_HORIZONTAL);

	// WHITE
	setStone(Location::getLoc(0, 0, x_size), S_KING ^ S_SIDE);
	setStone(Location::getLoc(1, 0, x_size), S_PAWN ^ S_SIDE);
	setStone(Location::getLoc(2, 0, x_size), S_BISHOP ^ S_SIDE);
	setStone(Location::getLoc(3, 0, x_size), S_GOLD ^ S_SIDE);
	setStone(Location::getLoc(4, 0, x_size), S_SILVER ^ S_SIDE);
	setStone(Location::getLoc(5, 0, x_size), S_KNIGHT ^ S_SIDE);
	setStone(Location::getLoc(6, 0, x_size), S_LANCE ^ S_SIDE);
	setStone(Location::getLoc(7, 0, x_size), S_PAWN ^ S_SIDE);
	//setStone(Location::getLoc(1, 8, x_size), S_KING ^ S_SIDE);

	for (int i = 2; i <= 6; ++i)
		setStone(Location::getLoc(i, 1, x_size), S_PAWN ^ S_SIDE);
	setStone(Location::getLoc(0, 1, x_size), S_PAWN ^ S_HORIZONTAL ^ S_SIDE);
	setStone(Location::getLoc(0, 2, x_size), S_ROOK ^ S_HORIZONTAL ^ S_SIDE);
	setStone(Location::getLoc(0, 3, x_size), S_GOLD ^ S_HORIZONTAL ^ S_SIDE);
	setStone(Location::getLoc(0, 4, x_size), S_SILVER ^ S_HORIZONTAL ^ S_SIDE);
	setStone(Location::getLoc(0, 5, x_size), S_KNIGHT ^ S_HORIZONTAL ^ S_SIDE);
	setStone(Location::getLoc(0, 6, x_size), S_LANCE ^ S_HORIZONTAL ^ S_SIDE);
	setStone(Location::getLoc(0, 7, x_size), S_PAWN ^ S_HORIZONTAL ^ S_SIDE);
	for (int i = 2; i <= 6; ++i)
		setStone(Location::getLoc(1, i, x_size), S_PAWN ^ S_SIDE ^ S_HORIZONTAL);

	//setStone(Location::getLoc(1, 1, x_size), S_KING ^ S_SIDE);
	//setStone(Location::getLoc(1, 2, x_size), S_LANCE);
	//setStone(Location::getLoc(1, 8, x_size), S_ROOK ^ S_HORIZONTAL);
	//setStone(Location::getLoc(3, 4, x_size), S_BISHOP);


//setStone(Location::getLoc(7, 0, x_size), S_KING);
//setStone(Location::getLoc(6, 0, x_size), S_SILVER);
//setStone(Location::getLoc(5, 0, x_size), S_SILVER);
//setStone(Location::getLoc(4, 0, x_size), S_SILVER);
//setStone(Location::getLoc(3, 0, x_size), S_SILVER);
//setStone(Location::getLoc(2, 0, x_size), S_SILVER);
//setStone(Location::getLoc(1, 0, x_size), S_SILVER);
//for(int i = 2; i <= 6; ++i)
//  setStone(Location::getLoc(i, 1, x_size), S_SILVER);
//setStone(Location::getLoc(8, 1, x_size), S_SILVER ^ S_HORIZONTAL);
//setStone(Location::getLoc(8, 2, x_size), S_SILVER ^ S_HORIZONTAL);
//setStone(Location::getLoc(8, 3, x_size), S_SILVER ^ S_HORIZONTAL);
//setStone(Location::getLoc(8, 4, x_size), S_SILVER ^ S_HORIZONTAL);
//setStone(Location::getLoc(8, 5, x_size), S_SILVER ^ S_HORIZONTAL);
//setStone(Location::getLoc(8, 6, x_size), S_SILVER ^ S_HORIZONTAL);
//setStone(Location::getLoc(8, 7, x_size), S_SILVER ^ S_HORIZONTAL);
//for(int i = 2; i <= 6; ++i)
//  setStone(Location::getLoc(7, i, x_size), S_SILVER ^ S_HORIZONTAL);

//// WHITE
//setStone(Location::getLoc(1, 8, x_size), S_KING ^ S_SIDE);
//setStone(Location::getLoc(2, 8, x_size), S_SILVER ^ S_SIDE);
//setStone(Location::getLoc(3, 8, x_size), S_SILVER ^ S_SIDE);
//setStone(Location::getLoc(4, 8, x_size), S_SILVER ^ S_SIDE);
//setStone(Location::getLoc(5, 8, x_size), S_SILVER ^ S_SIDE);
//setStone(Location::getLoc(6, 8, x_size), S_SILVER ^ S_SIDE);
//setStone(Location::getLoc(7, 8, x_size), S_SILVER ^ S_SIDE);

//for(int i = 2; i <= 6; ++i)
//  setStone(Location::getLoc(i, 7, x_size), S_SILVER ^ S_SIDE);
//setStone(Location::getLoc(0, 7, x_size), S_SILVER ^ S_HORIZONTAL ^ S_SIDE);
//setStone(Location::getLoc(0, 6, x_size), S_SILVER ^ S_HORIZONTAL ^ S_SIDE);
//setStone(Location::getLoc(0, 5, x_size), S_SILVER ^ S_HORIZONTAL ^ S_SIDE);
//setStone(Location::getLoc(0, 4, x_size), S_SILVER ^ S_HORIZONTAL ^ S_SIDE);
//setStone(Location::getLoc(0, 3, x_size), S_SILVER ^ S_HORIZONTAL ^ S_SIDE);
//setStone(Location::getLoc(0, 2, x_size), S_SILVER ^ S_HORIZONTAL ^ S_SIDE);
//setStone(Location::getLoc(0, 1, x_size), S_SILVER ^ S_HORIZONTAL ^ S_SIDE);
//for(int i = 2; i <= 6; ++i)
//  setStone(Location::getLoc(1, i, x_size), S_SILVER ^ S_SIDE ^ S_HORIZONTAL);

	repetition[0].insert(raw_pos_hash.hash0);
	repetition[1].insert(raw_pos_hash.hash1);
	checkRecord[nextPla].push_back(GameLogic::locAttacked(*this, kings[getOpp(nextPla)], nextPla));
	int repind = repetition[0].count(raw_pos_hash.hash0) < repetition[1].count(raw_pos_hash.hash1) ? 0 : 1;
	auto rephash = repind ? raw_pos_hash.hash1 : raw_pos_hash.hash0;
	first[repind][rephash] = checkRecord[nextPla].size() - 1;

	//raw_pos_hash ^= ZOBRIST_KOMA_HASH[P_BLACK][S_PAWN][++koma[P_BLACK][S_PAWN]];

	for (Player p = P_BLACK; p <= P_WHITE; ++p)
		KING_INIT_LOCATION[p] = kings[p];
	//playMoveAssumeLegal(Location::getLoc(8, 2, x_size), nextPla);
}

void Board::initHash() {
	if (IS_ZOBRIST_INITALIZED)
		return;
	Rand rand("Board::initHash()");
	auto nextHash = [&rand]() {
		uint64_t h0 = rand.nextUInt();
		uint64_t h1 = rand.nextUInt();
		return Hash128(h0, h1);
		};

	for (int i = 0; i < 4; i++)
		ZOBRIST_PLAYER_HASH[i] = nextHash();

	// Do this second so that the player and encore hashes are not
	// afffected by the size of the board we compile with.
	for (int i = 0; i < MAX_ARR_SIZE; i++) {
		for (Piece j = 0; j < NUM_BOARD_COLORS; j++) {
			if (j == C_EMPTY || j == C_WALL)
				ZOBRIST_BOARD_HASH[i][j] = Hash128();
			else
				ZOBRIST_BOARD_HASH[i][j] = nextHash();
		}
	}

	for (int i = 0; i < STAGE_NUM_EACH_PLA; i++) {
		ZOBRIST_STAGENUM_HASH[i] = nextHash();
		for (int j = 0; j < MAX_ARR_SIZE; j++)
			ZOBRIST_STAGELOC_HASH[j][i] = nextHash();
		ZOBRIST_STAGELOC_HASH[Board::NULL_LOC][i] = Hash128();
	}
	ZOBRIST_STAGENUM_HASH[0] = Hash128();

	for (Piece j = 0; j < 4; j++) {
		ZOBRIST_NEXTPLA_HASH[j] = nextHash();
	}

	ZOBRIST_MOVENUM_HASH[0] = Hash128();
	for (int i = 1; i < MAX_MOVE_NUM; i++) {
		ZOBRIST_MOVENUM_HASH[i] = nextHash();
	}

	// Reseed the random number generator so that these size hashes are also
	// not affected by the size of the board we compile with
	rand.init("Board::initHash() for ZOBRIST_SIZE hashes");
	for (int i = 0; i < MAX_LEN + 1; i++) {
		ZOBRIST_SIZE_X_HASH[i] = nextHash();
		ZOBRIST_SIZE_Y_HASH[i] = nextHash();
	}
	rand.init("Board::initHash() for KOMA hashes");
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < MAX_KOMA_SIZE; ++j) {
			ZOBRIST_KOMA_HASH[i][j][0] = Hash128();
			for (int k = 1; k < MAX_KOMA_SIZE; ++k)
				ZOBRIST_KOMA_HASH[i][j][k] = nextHash();
		}
	rand.init("Board::initHash() for REPETITION hashes");
	ZOBRIST_REPETITION_HASH[0] = Hash128();
	for (int i = 1; i <= MAX_REPETITION_NUM; ++i)
		ZOBRIST_REPETITION_HASH[i] = nextHash();
	ZOBRIST_REPLOSS_HASH[0] = Hash128();
	ZOBRIST_REPLOSS_HASH[1] = nextHash();
	IS_ZOBRIST_INITALIZED = true;
}

bool Board::isOnBoard(int x, int y) const {
	return Location::isOnBoard(x, y);
}
bool Board::isOnBoard(Loc loc) const {
	if (loc < 0 || loc >= MAX_ARR_SIZE)
		return false;
	int x = Location::getX(loc, x_size), y = Location::getY(loc, x_size);
	return isOnBoard(x, y);
}

void Board::applyMoveForAttackCheck(const Move& mv, Player pla, QuickCheckUndo& undo) const
{
	Board& self = const_cast<Board&>(*this);
	undo.isDrop = (mv.z != 0);
	undo.kingMoved = false;
	undo.from = mv.x;
	undo.to = mv.y;

	if (undo.isDrop)
	{
		undo.fromOld = self.colors[mv.x];
		Piece p = mv.z;
		if (mv.change) p ^= S_HORIZONTAL;
		if (pla == P_WHITE) p ^= S_SIDE;
		self.colors[mv.x] = p;
		return;
	}

	undo.fromOld = self.colors[mv.x];
	undo.toOld = self.colors[mv.y];
	Piece moved = undo.fromOld;
	if (mv.change) moved ^= S_PROMOTED;
	self.colors[mv.x] = C_EMPTY;
	self.colors[mv.y] = moved;

	if (getType(undo.fromOld) == S_KING)
	{
		undo.kingMoved = true;
		undo.oldKingLoc = self.kings[pla];
		self.kings[pla] = mv.y;
	}
}

void Board::undoMoveForAttackCheck(Player pla, const QuickCheckUndo& undo) const
{
	Board& self = const_cast<Board&>(*this);
	if (undo.isDrop)
	{
		self.colors[undo.from] = undo.fromOld;
		return;
	}

	self.colors[undo.from] = undo.fromOld;
	self.colors[undo.to] = undo.toOld;
	if (undo.kingMoved)
		self.kings[pla] = undo.oldKingLoc;
}

bool Board::wouldKingBeAttackedAfterMove(const Move& mv, Player pla) const
{
	QuickCheckUndo undo{};
	applyMoveForAttackCheck(mv, pla, undo);
	const Loc kingLoc = kings[pla];
	const bool attacked = GameLogic::locAttacked(*this, kingLoc, getOpp(pla));
	undoMoveForAttackCheck(pla, undo);
	return attacked;
}

//Check if moving here is illegal.
bool Board::genLegalMove(std::vector<Move>& mvs, Player pla, bool recursion)const
{
	std::vector<Loc> attackingPieces;
	bool attacked = GameLogic::getAttackingPiece(*this, kings[pla], getOpp(pla), attackingPieces);
	bool allowDropWhenChecked = true;
	if (attacked)
	{
		allowDropWhenChecked = attackingPieces.size() == 1;
		if (allowDropWhenChecked)
		{
			const Loc& loc = attackingPieces[0];
			const int& x = Location::getX(loc, x_size), & y = Location::getY(loc, x_size);
			const int& kx = Location::getX(kings[pla], x_size), & ky = Location::getY(kings[pla], x_size);
			const int& dx = abs(kx - x), & dy = abs(ky - y);
			allowDropWhenChecked = dx > 1 && (!dy || dx == dy) || !dx && dy > 1;
		}
	}
	mvs.clear();
	if (!recursion && mvs.capacity() < 256)
		mvs.reserve(256);
	if (winner != C_WALL)return false;
	bool hasPawnV[MAX_LEN]{ false }, hasPawnH[MAX_LEN]{ false };
	bool maybePinned[MAX_ARR_SIZE]{ false };
	std::vector<Move> tmp_reply_moves;
	if (!recursion)
		tmp_reply_moves.reserve(1);
	auto checkAttacked = [&](const Move& m) {
		return wouldKingBeAttackedAfterMove(m, pla);
		};
	{
		const Loc kingLoc = kings[pla];
		const int kx = Location::getX(kingLoc, x_size);
		const int ky = Location::getY(kingLoc, x_size);
		constexpr int rayDx[8]{ 1,-1,0,0,1,1,-1,-1 };
		constexpr int rayDy[8]{ 0,0,1,-1,1,-1,1,-1 };
		for (int d = 0; d < 8; ++d)
		{
			for (int nx = kx + rayDx[d], ny = ky + rayDy[d]; isOnBoard(nx, ny); nx += rayDx[d], ny += rayDy[d])
			{
				Loc l = Location::getLoc(nx, ny, x_size);
				Piece pc = colors[l];
				if (pc == C_EMPTY)continue;
				if (getSide(pc) == pla)
					maybePinned[l] = true;
				break;
			}
		}
	}
	for (int i = 0; i < x_size; ++i)
		for (int j = 0; j < y_size; ++j)
		{
			if (!isOnBoard(i, j))continue;
			Piece pc = colors[Location::getLoc(i, j, x_size)];
			if (getSide(pc) == pla && getType(pc) == S_PAWN && !getPromoted(pc))
			{
				if (getHorizontal(pc))hasPawnH[j] = true;
				else hasPawnV[i] = true;
			}
		}

	for (int i = 0; i < x_size; ++i)
		for (int j = 0; j < y_size; ++j)
		{
			if (!isOnBoard(i, j))continue;
			Loc from = Location::getLoc(i, j, x_size);
			Piece pc = colors[from];
			if (pc != C_EMPTY && getSide(pc) != pla)continue;
			auto must = [this, pla](int x, int y, Piece pc)
				{
					if (pla == P_WHITE)x = x_size - 1 - x, y = y_size - 1 - y;
					int tp = getType(pc);
					if (getPromoted(pc) || tp == S_KING || tp == S_GOLD)return -1;
					if (getHorizontal(pc))
					{
						if (tp == S_KNIGHT)return x < 2 ? 1 : 0;
						if (tp == S_PAWN || tp == S_LANCE)return isOnBoard(x - 1, y) ? 0 : 1;
						return 0;
					}
					else
					{
						if (tp == S_KNIGHT)return y < 2 ? 1 : 0;
						if (tp == S_PAWN || tp == S_LANCE)return isOnBoard(x, y - 1) ? 0 : 1;
						return 0;
					}
				};
			if (pc != C_EMPTY)
			{
				const bool needAttackCheck = attacked || getType(pc) == S_KING || maybePinned[from];
				auto tp = getType(pc);
				struct Dir { int dx; int dy; bool slide; };
				Dir dirs[8];
				int dir_count = 0;
				auto push_dir = [&dirs, &dir_count](int dx, int dy, bool slide) {
					dirs[dir_count++] = { dx, dy, slide };
					};

				switch (tp)
				{
				case S_PAWN:
					push_dir(0, -1, false);
					break;
				case S_GOLD:
					push_dir(-1, -1, false); push_dir(-1, 0, false); push_dir(0, -1, false);
					push_dir(0, 1, false); push_dir(1, -1, false); push_dir(1, 0, false);
					break;
				case S_BISHOP:
					push_dir(1, 1, true); push_dir(1, -1, true); push_dir(-1, 1, true); push_dir(-1, -1, true);
					break;
				case S_ROOK:
					push_dir(1, 0, true); push_dir(-1, 0, true); push_dir(0, 1, true); push_dir(0, -1, true);
					break;
				case S_LANCE:
					push_dir(0, -1, true);
					break;
				case S_KNIGHT:
					push_dir(-1, -2, false); push_dir(1, -2, false);
					break;
				case S_KING:
					push_dir(1, 1, false); push_dir(1, 0, false); push_dir(1, -1, false); push_dir(-1, 1, false);
					push_dir(-1, 0, false); push_dir(-1, -1, false); push_dir(0, 1, false); push_dir(0, -1, false);
					break;
				case S_SILVER:
					push_dir(-1, -1, false); push_dir(-1, 1, false); push_dir(0, -1, false);
					push_dir(1, -1, false); push_dir(1, 1, false);
					break;
				default:
					break;
				}

				if (getPromoted(pc))
				{
					if (tp == S_PAWN || tp == S_LANCE || tp == S_SILVER || tp == S_KNIGHT)
					{
						dir_count = 0;
						push_dir(-1, -1, false); push_dir(-1, 0, false); push_dir(0, -1, false);
						push_dir(0, 1, false); push_dir(1, -1, false); push_dir(1, 0, false);
					}
					else if (tp == S_ROOK)
					{
						push_dir(1, 1, false); push_dir(1, -1, false); push_dir(-1, 1, false); push_dir(-1, -1, false);
					}
					else if (tp == S_BISHOP)
					{
						push_dir(1, 0, false); push_dir(-1, 0, false); push_dir(0, 1, false); push_dir(0, -1, false);
					}
				}

				for (int k = 0; k < dir_count; ++k)
				{
					int dx = dirs[k].dx;
					int dy = dirs[k].dy;
					if (getHorizontal(pc))
					{
						const int tmp = dx;
						dx = dy;
						dy = -tmp;
					}
					if (pla == P_WHITE)
					{
						dx = -dx;
						dy = -dy;
					}

					for (int nx = i + dx, ny = j + dy; isOnBoard(nx, ny); nx += dx, ny += dy)
					{
						Loc to = Location::getLoc(nx, ny, x_size);
						if (getSide(colors[to]) == pla)break;
						Move mv1(from, to), mv2(from, to, 0, true);
						int ms = must(nx, ny, pc);
						if (ms <= 0)
							if (!needAttackCheck || !checkAttacked(mv1))
								if (recursion)return true;
								else mvs.push_back(mv1);
						bool inArea = false;
						if (getHorizontal(pc))
							if (pla == P_BLACK)inArea = i < 3 || nx < 3;
							else inArea = i > 5 || nx > 5;
						else
							if (pla == P_BLACK)inArea = j < 3 || ny < 3;
							else inArea = j > 5 || ny > 5;
						if (ms >= 0 && inArea)
							if (!needAttackCheck || !checkAttacked(mv2))
								if (recursion)return true;
								else mvs.push_back(mv2);

						if (!dirs[k].slide)break;
						if (colors[to] != C_EMPTY)break;
					}
				}
				continue;
			}
			if (attacked && !allowDropWhenChecked)continue;
			for (Piece p = S_GOLD; p <= S_PAWN; ++p)
			{
				if (koma[pla][p] <= 0)continue;
				Move mv1(from, 0, p);
				if (must(i, j, p) != 1)
					if (p != S_PAWN)
					{
						if (!attacked || !checkAttacked(mv1))
							if (recursion)return true;
							else mvs.push_back(mv1);
					}
					else if (!hasPawnV[i])
					{
						bool can = true;
						if (attacked && checkAttacked(mv1))can = false;
						if (can && !recursion)
						{
							Board bd(*this, true);
							bd.playMoveAssumeLegal(Move(from, 0, p), pla);
							tmp_reply_moves.clear();
							can = bd.genLegalMove(tmp_reply_moves, getOpp(pla), true);
						}
						if (can)
							if (recursion)return true;
							else mvs.push_back(mv1);
					}
				Move mv2(from, 0, p, true);
				if (must(i, j, p ^ S_HORIZONTAL) != 1)
					if (p != S_PAWN)
					{
						if (!attacked || !checkAttacked(mv2))
							if (recursion)return true;
							else mvs.push_back(mv2);
					}
					else if (!hasPawnH[j])
					{
						bool can = true;
						if (attacked && checkAttacked(mv2))can = false;
						if (can && !recursion)
						{
							Board bd(*this, true);
							bd.playMoveAssumeLegal(Move(from, 0, p, true), pla);
							tmp_reply_moves.clear();
							can = bd.genLegalMove(tmp_reply_moves, getOpp(pla), true);
						}
						if (can)
							if (recursion)return true;
							else mvs.push_back(mv2);
					}
			}
		}
	return !mvs.empty();
}
std::vector<Move> Board::genLegalMove()const
{
	std::vector<Move> ret;
	genLegalMove(ret, nextPla);
	return ret;
}
void Board::genLegalPolicy(std::vector<size_t>& ret)const
{
	std::vector<Move> mvs;
	mvs.reserve(256);
	genLegalMove(mvs, nextPla);
	ret.resize(mvs.size());
	using namespace Location;
	if (nextPla == P_BLACK)
	{
		for (size_t i = 0; i < mvs.size(); ++i)
			ret[i] = movePolicy[mvs[i]];
	}
	else
	{
		for (size_t i = 0; i < mvs.size(); ++i)
		{
			Move mv = mvs[i];
			ret[i] = movePolicy[mv.rot()];
		}
	}
}
std::vector<size_t> Board::genLegalPolicy()const
{
	std::vector<size_t> ret;
	genLegalPolicy(ret);
	return ret;
}
std::vector<int> Board::get_legal_moves()
{
	std::vector<Move> mvs;
	std::vector<int> ret(get_action_size(), 0);
	//cout << ret.size() << endl;

	//printBoard(cout, 0);
	genLegalMove(mvs, nextPla);
	if (nextPla == P_WHITE)
		for (Move& mv : mvs)mv.rot();
	for (const Move& mv : mvs)
	{
		assert(Location::movePolicy[mv] >= 0);
		using namespace Location;
		if (Location::movePolicy[mv] == -1)
			cout << "played:" << '(' << getX(mv.x, 9) + 1 << ',' << getY(mv.x, 9) + 1 << ")->(" << getX(mv.y, 9) + 1 << ',' << getY(mv.y, 9) + 1 << ") " << (int)mv.z << mv.change << endl,
			printBoard(cout, 0);
		//cout << mv << endl;
		ret[Location::movePolicy[mv]] = 1;
	}
	return ret;
}
/*bool Board::isLegal(Move mv, Player pla) const {
  const Board& board = *this;
  if(pla != board.nextPla) {
	std::cerr << "Error next player ";
	return false;
  }
  if (mv.z)
	  return GameLogic::checkLegal(board, Board::PASS_LOC, mv.z, mv.x, pla);
  // if((board.isOnBoard(loc) || (y == board.y_size * 2)) && getSide(board.colors[loc]) == pla)
  else
	return GameLogic::checkLegal(board, board.locChosen[1], board.colors[board.locChosen[1]], loc, pla);
  else
	ASSERT_UNREACHABLE;
  return false;
}
*/
bool Board::isEmpty() const {
	for (int y = 0; y < y_size; y++) {
		for (int x = 0; x < x_size; x++) {
			Loc loc = Location::getLoc(x, y, x_size);
			if (colors[loc] != C_EMPTY)
				return false;
		}
	}
	return true;
}

int Board::numStonesOnBoard() const {
	int num = 0;
	for (int y = 0; y < y_size; y++) {
		for (int x = 0; x < x_size; x++) {
			Loc loc = Location::getLoc(x, y, x_size);
			if (colors[loc] != C_EMPTY)
				num += 1;
		}
	}
	return num;
}

int Board::numPlaStonesOnBoard(Player pla) const {
	int num = 0;
	for (int y = 0; y < y_size; y++) {
		for (int x = 0; x < x_size; x++) {
			Loc loc = Location::getLoc(x, y, x_size);
			if (colors[loc] == pla)
				num += 1;
		}
	}
	return num;
}
int Board::get_action_size()
{
	return Location::policy.size();
}
bool Board::setStone(Loc loc, Piece color)
{
	if (loc < 0 || loc >= MAX_ARR_SIZE || colors[loc] == C_WALL)
		return false;
	Player pla = getSide(color);
	Piece colorOld = colors[loc];
	colors[loc] = color;
	raw_pos_hash ^= ZOBRIST_BOARD_HASH[loc][colorOld];
	raw_pos_hash ^= ZOBRIST_BOARD_HASH[loc][color];
	if (getType(color) == S_KING)
		kings[getSide(color)] = loc;
	Piece tp = getType(colorOld);
	if (color != C_EMPTY && colorOld != C_EMPTY && getSide(colorOld) == getOpp(pla) && tp != S_KING) {
		raw_pos_hash ^= ZOBRIST_KOMA_HASH[pla][tp][koma[pla][tp]];
		raw_pos_hash ^= ZOBRIST_KOMA_HASH[pla][tp][++koma[pla][tp]];
	}
	return true;
}
/*bool Board::setStones(std::vector<Move> placements) {
  std::set<Loc> locs;
  for(const Move& placement: placements) {
	if(locs.find(placement.loc) != locs.end())
	  return false;
	locs.insert(placement.loc);
  }
  // First empty out all locations that we plan to set.
  // This guarantees avoiding any intermediate liberty issues.
  for(const Move& placement: placements) {
	bool suc = setStone(placement.loc, C_EMPTY);
	if(!suc)
	  return false;
  }
  // Now set all the stones we wanted.
  for(const Move& placement: placements) {
	bool suc = setStone(placement.loc, placement.pla);
	if(!suc)
	  return false;
  }
  return true;
}*/

//Plays the specified move, assuming it is legal.
void Board::playMoveAssumeLegal(Move mv, Player pla) {
	if (pla != nextPla) {
		std::cerr << "Error next player ";
	}
	if (winner != C_WALL) {
		cout << moveToPolicy(mv) << endl;
		std::cout << "Game Has Ended" << endl;
		//printBoard(std::cout, 0);
		return;
	}
	if (mv.z)
	{
		Piece p = mv.z;
		if (mv.change)p ^= S_HORIZONTAL;
		if (pla == P_WHITE)p ^= S_SIDE;
		setStone(mv.x, p);
		raw_pos_hash ^= ZOBRIST_KOMA_HASH[pla][mv.z][koma[pla][mv.z]];
		raw_pos_hash ^= ZOBRIST_KOMA_HASH[pla][mv.z][--koma[pla][mv.z]];
	}
	else
	{
		/*
		if (getType(colors[mv.y]) != S_KING)
			++koma[getOpp(pla)][getType(colors[mv.y])];
		else ASSERT_UNREACHABLE;*/
		Piece p = colors[mv.x];
		if (mv.change)p ^= S_PROMOTED;
		if (getType(colors[mv.y]) == S_KING)
		{
			cout << *this;
			cout << "ERROR!!! KING EATEN!!!" << endl;
			cout << "BY PIECE" << PlayerIO::colorToChar(colors[mv.x]) << endl;
			cout << mv.x << ' ' << mv.y << endl;
			cout << (int)getSide(p) << ' ' << (int)pla << endl;
			winner = pla;
		}
		setStone(mv.y, p);
		setStone(mv.x, C_EMPTY);
	}

	//raw_pos_hash ^= ZOBRIST_MOVENUM_HASH[movenum];
	++movenum;
	//raw_pos_hash ^= ZOBRIST_MOVENUM_HASH[movenum];

	raw_pos_hash ^= ZOBRIST_NEXTPLA_HASH[nextPla];
	nextPla = getOpp(nextPla);
	raw_pos_hash ^= ZOBRIST_NEXTPLA_HASH[nextPla];
	// raw_pos_hash ^= ZOBRIST_REPETITION_HASH[repcount];
	//if (GameLogic::locAttacked(*this, kings[pla], getOpp(pla)))
	//{
	//	cout << "ERROR!!! KING IN CHECK!!!" << endl;
	//	//throw "ERROR!!! KING IN CHECK!!!";
	//	cout << *this;
	//	winner = getOpp(pla);
	//	return;
	//}
	//repetition[0][1] = 1;
	//cout << repetition[0].size() << endl;
	attacked = GameLogic::locAttacked(*this, kings[getOpp(pla)], pla);
	if (!dummy)
	{
		bool willDraw = false;
		int repind = repetition[0].count(raw_pos_hash.hash0) < repetition[1].count(raw_pos_hash.hash1) ? 0 : 1;
		auto rephash = repind ? raw_pos_hash.hash1 : raw_pos_hash.hash0;
		checkRecord[pla].push_back(attacked + checkRecord[pla].back());
		if (!first[repind][rephash])
			first[repind][rephash] = checkRecord[pla].size() - 1;
		else {
			const int& ff = first[repind][rephash];
			if (checkRecord[pla].back() - checkRecord[pla][ff] != checkRecord[pla].size() - 1 - ff)
				willDraw = true;
		}
		if (repetition[repind].count(rephash))
			winner = willDraw ? C_EMPTY : getOpp(pla);
		else
		{
			repetition[0].insert(raw_pos_hash.hash0);
			repetition[1].insert(raw_pos_hash.hash1);
		}
	}
	if (winner == C_WALL)
		if (kings[pla] == KING_INIT_LOCATION[getOpp(pla)])
			winner = pla;
	//std::vector<Move> tmp;
	//cout << "Moving:" << endl;
	//cout << Location::getX(loc, x_size) << ' ' << Location::getY(loc, x_size) << ' ' << movenum << ' ' << winner << endl;
}
void Board::playPolicy(int p)
{
	Move mv = Location::policy[p];
	if (nextPla == P_WHITE)mv.rot();
	playMoveAssumeLegal(mv, nextPla);
}
size_t Board::moveToPolicy(Move mv)
{
	if (nextPla == P_WHITE)mv.rot();
	return Location::movePolicy[mv];
}
Hash128 Board::getSitHash(Player pla) const {
	Hash128 h = getPosHash();
	h ^= Board::ZOBRIST_PLAYER_HASH[pla];
	return h;
}

int Location::distance(Loc loc0, Loc loc1, int x_size) {
	int dx = getX(loc1, x_size) - getX(loc0, x_size);
	int dy = (loc1 - loc0 - dx) / (x_size + 1);
	return (dx >= 0 ? dx : -dx) + (dy >= 0 ? dy : -dy);
}

int Location::euclideanDistanceSquared(Loc loc0, Loc loc1, int x_size) {
	int dx = getX(loc1, x_size) - getX(loc0, x_size);
	int dy = (loc1 - loc0 - dx) / (x_size + 1);
	return dx * dx + dy * dy;
}

//TACTICAL STUFF--------------------------------------------------------------------


void Board::checkConsistency() const {
	const string errLabel = string("Board::checkConsistency(): ");


	vector<Loc> buf;
	Hash128 tmp_pos_hash = ZOBRIST_SIZE_X_HASH[x_size] ^ ZOBRIST_SIZE_Y_HASH[y_size];
	for (Loc loc = 0; loc < MAX_ARR_SIZE; loc++) {
		int x = Location::getX(loc, x_size);
		int y = Location::getY(loc, x_size);
		if (x < 0 || x >= x_size || y < 0 || y >= y_size) {
			if (colors[loc] != C_WALL)
				throw errLabel + "Non-WALL value outside of board legal area";
		}
		else if (colors[loc] != C_WALL) {
			tmp_pos_hash ^= ZOBRIST_BOARD_HASH[loc][colors[loc]];
			tmp_pos_hash ^= ZOBRIST_BOARD_HASH[loc][C_EMPTY];
		}
	}

	//tmp_pos_hash ^= ZOBRIST_MOVENUM_HASH[movenum];

	tmp_pos_hash ^= ZOBRIST_NEXTPLA_HASH[nextPla];

	for (int pla = P_BLACK; pla <= P_WHITE; ++pla) {
		for (int pc = S_KING + 1; pc <= S_PAWN; ++pc)
			tmp_pos_hash ^= ZOBRIST_KOMA_HASH[pla][pc][koma[pla][pc]];
	}

	if (raw_pos_hash != tmp_pos_hash) {
		std::cout << "NextPla=" << int(nextPla) << std::endl;
		cout << *this;
		//std::cout << (raw_pos_hash ^ tmp_pos_hash) << endl;
		//printBoard(cout, *this, 0, NULL);
		throw errLabel + "Raw Pos hash does not match expected";
	}
	if (getPosHash() != getPosHash(tmp_pos_hash)) {
		std::cout << "NextPla=" << int(nextPla) << std::endl;
		throw errLabel + "Pos hash does not match expected";
	}


	short tmpAdjOffsets[8];
	Location::getAdjacentOffsets(tmpAdjOffsets, x_size);
	for (int i = 0; i < 8; i++)
		if (tmpAdjOffsets[i] != adj_offsets[i])
			throw errLabel + "Corrupted adj_offsets array";
}

bool Board::isEqualForTesting(const Board& other) const {
	checkConsistency();
	other.checkConsistency();
	if (x_size != other.x_size)
		return false;
	if (y_size != other.y_size)
		return false;
	if (getPosHash() != other.getPosHash())
		return false;
	for (int i = 0; i < MAX_ARR_SIZE; i++) {
		if (colors[i] != other.colors[i])
			return false;
	}
	//We don't require that the chain linked lists are in the same order.
	//Consistency check ensures that all the linked lists are consistent with colors array, which we checked.
	return true;
}



//IO FUNCS------------------------------------------------------------------------------------------

string PlayerIO::colorToChar(Piece c)
{
	string s;
	Player pla = getSide(c);
	if (c == C_WALL)
		return "#  ";
	else if (c == C_EMPTY)
		return ".  ";
	if (pla == P_BLACK)
		s += 'b';
	else if (pla == P_WHITE)
		s += 'w';
	switch (getType(c)) {
	case S_KING:
		s += "K";
		break;
	case S_BISHOP:
		s += "B";
		break;
	case S_PAWN:
		s += "P";
		break;
	case S_KNIGHT:
		s += "N";
		break;
	case S_GOLD:
		s += "G";
		break;
	case S_SILVER:
		s += "S";
		break;
	case S_LANCE:
		s += "L";
		break;
	case S_ROOK:
		s += "R";
	default:
		break;
	}
	if (getPromoted(c))
		s += getHorizontal(c) ? '-' : '+';
	else
		s += getHorizontal(c) ? '=' : '~';
	return s;
}

string PlayerIO::playerToString(Player c)
{
	switch (c) {
	case P_BLACK: return "Black";
	case P_WHITE: return "White";
	case C_EMPTY: return "Empty";
	default:  return "Wall";
	}
}

string PlayerIO::playerToStringShort(Player c)
{
	switch (c) {
	case P_BLACK: return "B";
	case P_WHITE: return "W";
	case C_EMPTY: return "E";
	default:  return "";
	}
}

void Board::printBoard(ostream& out, const vector<Move>* hist)const {
	//if (hist != NULL)
	//    out << "MoveNum: " << hist->size() << " ";
	out << "WholeMoveNum: " << movenum << '\n';
	//out << "HASH: " << raw_pos_hash << "\n";
	const char* pcs[]{ "金", "银","桂","香","飞","角","步" };
	out << "winner: " << winner << '\n';
	out << "Black:\n";
	for (int i = S_KING + 1; i <= S_PAWN; ++i) {
		out << pcs[i - S_KING - 1] << ':' << koma[P_BLACK][i] << ' ';
	}
	out << '\n';

	out << "White:\n";
	for (int i = S_KING + 1; i <= S_PAWN; ++i) {
		out << pcs[i - S_KING - 1] << ':' << koma[P_WHITE][i] << ' ';
	}
	out << '\n';
	out << "   A   B   C   D   E   F   G   H   I\n";
	for (int j = 0; j < y_size; ++j)
	{
		out << 9 - j << "  ";
		for (int i = 0; i < x_size; ++i)
			out << PlayerIO::colorToChar(colors[Location::getLoc(i, j, x_size)]) << ' ';
		out << '\n';
	}
	out.flush();
	//auto bd(board);
	//int repind = bd.repetition[0][board.raw_pos_hash.hash0] < bd.repetition[1][board.raw_pos_hash.hash1] ? 0 : 1;
	//auto rephash = repind ? board.raw_pos_hash.hash1 : board.raw_pos_hash.hash0;
	//bool showCoords = board.x_size <= 50 && board.y_size <= 50;
	//if (showCoords) {
	//    const char* xChar = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
	//    //out << "   ";
	//    for (int x = 0; x < board.x_size; x++) {
	//        if (x <= 25) {
	//            out << "   ";
	//            out << xChar[x];
	//        }
	//        else {
	//            out << "A" << xChar[x - 26];
	//        }
	//    }
	//    out << "\n";
	//}

	//for (int y = board.y_size - 1; y >= 0; --y) {
	//    if (showCoords) {
	//        char buf[16];
	//        sprintf(buf, "%2d", y + 1);
	//        out << buf << ' ';
	//    }
	//    for (int x = 0; x < board.x_size; ++x) {
	//        Loc loc = Location::getLoc(x, y, board.x_size);
	//        string s = PlayerIO::colorToChar(board.colors[loc]);
	//        if (board.colors[loc] == C_EMPTY && markLoc == loc)
	//            out << '@';
	//        else {
	//            char buf[16];
	//            sprintf(buf, "%-3s", s.c_str());
	//            out << buf;
	//        }

	//        bool histMarked = false;
	//        if (hist != NULL) {
	//            size_t start = hist->size() >= 3 ? hist->size() - 3 : 0;
	//            for (size_t i = 0; start + i < hist->size(); i++) {
	//                if ((*hist)[start + i].loc == loc) {
	//                    // out << (1 + i);
	//                    //histMarked = true;
	//                    break;
	//                }
	//            }
	//        }

	//        if (x < board.x_size - 1 && !histMarked)
	//            out << ' ';
	//    }
	//    out << "\n";
	//}
	//out << "\n";
}

ostream& operator<<(ostream& out, const Board& board) {
	board.printBoard(out, NULL);
	return out;
}
/*
bool PlayerIO::tryParsePlayer(const string& s, Player& pla) {
  string str = Global::toLower(s);
  if(str == "black" || str == "b") {
	pla = P_BLACK;
	return true;
  }
  else if(str == "white" || str == "w") {
	pla = P_WHITE;
	return true;
  }
  return false;
}

Player PlayerIO::parsePlayer(const string& s) {
  Player pla = C_EMPTY;
  bool suc = tryParsePlayer(s,pla);
  if(!suc)
	throw "Could not parse player: " + s;
  return pla;
}

string Location::toStringMach(Loc loc, int x_size)
{
  if(loc == Board::PASS_LOC)
	return string("pass");
  if(loc == Board::NULL_LOC)
	return string("null");
  char buf[128];
  sprintf(buf,"(%d,%d)",getX(loc,x_size),getY(loc,x_size));
  return string(buf);
}

string Location::toString(Loc loc, int x_size, int y_size) {
  if(x_size > 26 * 26)
	return toStringMach(loc, x_size);
  if(loc == Board::PASS_LOC)
	return string("pass");
  if(loc == Board::NULL_LOC)
	return string("null");
  const char* xChar = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  int x = getX(loc, x_size);
  int y = getY(loc, x_size);
  if(x >= x_size || x < 0 || y < 0 || y >= y_size)
	return toStringMach(loc, x_size);

  char buf[128];
  if(x <= 25)
	sprintf(buf, "%c%d", xChar[x], y + 1);
  else
	sprintf(buf, "%c%c%d", xChar[x / 26 - 1], xChar[x % 26], y + 1);
  return string(buf);
}

string Location::toString(Loc loc, const Board& b) {
  return toString(loc,b.x_size,b.y_size);
}

string Location::toStringMach(Loc loc, const Board& b) {
  return toStringMach(loc,b.x_size);
}

static bool tryParseLetterCoordinate(char c, int& x) {
  if(isupper(c))
	x = c - 'A';
  else if(islower(c))
	x = c - 'a';
  else
	return false;
  return true;
}

bool Location::tryOfString(const string& str, int x_size, int y_size, Loc& result) {
  string s = Global::trim(str);
  if(s.length() < 2)
	return false;
  if(Global::isEqualCaseInsensitive(s, string("pass")) || Global::isEqualCaseInsensitive(s, string("pss"))) {
	result = Board::PASS_LOC;
	return true;
  }
  if(s[0] == '(') {
	if(s[s.length() - 1] != ')')
	  return false;
	s = s.substr(1, s.length() - 2);
	vector<string> pieces = Global::split(s, ',');
	if(pieces.size() != 2)
	  return false;
	int x;
	int y;
	bool sucX = Global::tryStringToInt(pieces[0], x);
	bool sucY = Global::tryStringToInt(pieces[1], y);
	if(!sucX || !sucY)
	  return false;
	result = Location::getLoc(x, y, x_size);
	return true;
  } else {
	int x;
	if(!tryParseLetterCoordinate(s[0], x))
	  return false;

	// Extended format
	if((s[1] >= 'A' && s[1] <= 'Z') || (s[1] >= 'a' && s[1] <= 'z')) {
	  int x1;
	  if(!tryParseLetterCoordinate(s[1], x1))
		return false;
	  x = (x + 1) * 26 + x1;
	  s = s.substr(2, s.length() - 2);
	} else {
	  s = s.substr(1, s.length() - 1);
	}

	int y;
	bool sucY = Global::tryStringToInt(s, y);
	if(!sucY)
	  return false;
	--y;
	if(x < 0 || y < 0 || x >= x_size || y >= y_size)
	  return false;
	result = Location::getLoc(x, y, x_size);
	return true;
  }
}

bool Location::tryOfStringAllowNull(const string& str, int x_size, int y_size, Loc& result) {
  if(str == "null") {
	result = Board::NULL_LOC;
	return true;
  }
  return tryOfString(str, x_size, y_size, result);
}

bool Location::tryOfString(const string& str, const Board& b, Loc& result) {
  return tryOfString(str,b.x_size,b.y_size,result);
}

bool Location::tryOfStringAllowNull(const string& str, const Board& b, Loc& result) {
  return tryOfStringAllowNull(str,b.x_size,b.y_size,result);
}

Loc Location::ofString(const string& str, int x_size, int y_size) {
  Loc result;
  if(tryOfString(str,x_size,y_size,result))
	return result;
  throw StringError("Could not parse board location: " + str);
}

Loc Location::ofStringAllowNull(const string& str, int x_size, int y_size) {
  Loc result;
  if(tryOfStringAllowNull(str,x_size,y_size,result))
	return result;
  throw StringError("Could not parse board location: " + str);
}

Loc Location::ofString(const string& str, const Board& b) {
  return ofString(str,b.x_size,b.y_size);
}


Loc Location::ofStringAllowNull(const string& str, const Board& b) {
  return ofStringAllowNull(str,b.x_size,b.y_size);
}

vector<Loc> Location::parseSequence(const string& str, const Board& board) {
  vector<string> pieces = Global::split(Global::trim(str),' ');
  vector<Loc> locs;
  for(size_t i = 0; i<pieces.size(); i++) {
	string piece = Global::trim(pieces[i]);
	if(piece.length() <= 0)
	  continue;
	locs.push_back(Location::ofString(piece,board));
  }
  return locs;
}

string Board::toStringSimple(const Board& board, char lineDelimiter) {
  string s;
  for(int y = 0; y < board.y_size; y++) {
	for(int x = 0; x < board.x_size; x++) {
	  Loc loc = Location::getLoc(x,y,board.x_size);
	  s += PlayerIO::colorToChar(board.colors[loc]);
	}
	s += lineDelimiter;
  }
  return s;
}

Board Board::parseBoard(int xSize, int ySize, const string& s) {
  return parseBoard(xSize,ySize,s,'\n');
}

Board Board::parseBoard(int xSize, int ySize, const string& s, char lineDelimiter) {
  Board board(xSize, ySize);
  vector<string> lines = Global::split(Global::trim(s), lineDelimiter);

  // Throw away coordinate labels line if it exists
  if(lines.size() == ySize + 1 && Global::isPrefix(lines[0], "A"))
	lines.erase(lines.begin());

  if(lines.size() != ySize)
	throw StringError("Board::parseBoard - string has different number of board rows than ySize");

  for(int y = 0; y < ySize; y++) {
	string line = Global::trim(lines[y]);
	// Throw away coordinates if they exist
	size_t firstNonDigitIdx = 0;
	while(firstNonDigitIdx < line.length() && Global::isDigit(line[firstNonDigitIdx]))
	  firstNonDigitIdx++;
	line.erase(0, firstNonDigitIdx);
	line = Global::trim(line);

	if(line.length() != xSize && line.length() != 2 * xSize - 1)
	  throw StringError("Board::parseBoard - line length not compatible with xSize");

	for(int x = 0; x < xSize; x++) {
	  char c;
	  if(line.length() == xSize)
		c = line[x];
	  else
		c = line[x * 2];

	  Loc loc = Location::getLoc(x, y, board.x_size);
	  if(c == '.' || c == ' ' || c == '*' || c == ',' || c == '`')
		continue;
	  else if(c == 'o' || c == 'O') {
		bool suc = board.setStone(loc, P_WHITE);
		if(!suc)
		  throw StringError(string("Board::parseBoard - zero-liberty group near ") + Location::toString(loc, board));
	  } else if(c == 'x' || c == 'X') {
		bool suc = board.setStone(loc, P_BLACK);
		if(!suc)
		  throw StringError(string("Board::parseBoard - zero-liberty group near ") + Location::toString(loc, board));
	  } else
		throw StringError(string("Board::parseBoard - could not parse board character: ") + c);
	}
  }
  return board;
}

nlohmann::json Board::toJson(const Board& board) {
  nlohmann::json data;
  data["xSize"] = board.x_size;
  data["ySize"] = board.y_size;
  data["stones"] = Board::toStringSimple(board,'|');
  return data;
}

Board Board::ofJson(const nlohmann::json& data) {
  int xSize = data["xSize"].get<int>();
  int ySize = data["ySize"].get<int>();
  Board board = Board::parseBoard(xSize,ySize,data["stones"].get<string>(),'|');
  return board;
}*/

