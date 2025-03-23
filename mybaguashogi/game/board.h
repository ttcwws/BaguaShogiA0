/*
 * board.h
 * Originally from an unreleased project back in 2010, modified since.
 * Authors: brettharrison (original), David Wu (original and later modifications).
 */
#pragma once
#ifndef GAME_BOARD_H_
#define GAME_BOARD_H_

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <iostream>
#include <string>
#include "../core/rand.h"
using std::endl;
using std::cout;
using std::string;

#ifndef COMPILE_MAX_BOARD_LEN 
#define COMPILE_MAX_BOARD_LEN 9
#endif

//how many stages in each move
//eg: Chess has 2 stages: moving which piece, and where to place.
static const int STAGE_NUM_EACH_PLA = 3;

//max moves num of a game
static const int MAX_MOVE_NUM = 100 * COMPILE_MAX_BOARD_LEN * COMPILE_MAX_BOARD_LEN;
static const int MAX_BIG_MOVE_NUM = 1000;
static const int MAX_KOMA_SIZE = 9;
static const int MAX_KOMA_NUM = 25;
static const int MAX_REPETITION_NUM = 4;

//TYPES AND CONSTANTS-----------------------------------------------------------------

struct Board;

//Player
typedef uint8_t Player;
typedef uint8_t Piece;
static constexpr Player P_BLACK = 1;
static constexpr Player P_WHITE = 2;

static constexpr Piece S_KING = 1;
static constexpr Piece S_GOLD = 2;
static constexpr Piece S_SILVER = 3;
static constexpr Piece S_KNIGHT = 4;
static constexpr Piece S_LANCE = 5;
static constexpr Piece S_ROOK = 6;
static constexpr Piece S_BISHOP = 7;
static constexpr Piece S_PAWN = 8;
static constexpr Piece S_NUM_PIECES = 8;
static constexpr Piece S_HORIZONTAL = 0x10;
static constexpr Piece S_PROMOTED = 0x20;
static constexpr Piece S_SIDE = 0x40;
static constexpr Piece S_TAKEPIECE = 0xF;
static constexpr int BOARD_AREA = COMPILE_MAX_BOARD_LEN * COMPILE_MAX_BOARD_LEN;



//Color of a point on the board
static constexpr Piece C_EMPTY = 0;
static constexpr Piece C_WALL = 0x7F;
static constexpr int NUM_BOARD_COLORS = 128;

static inline Player getOpp(Player c) {
  return c ^ 3;
}
static inline Player getSide(Piece c) {
  if(c == C_EMPTY || c == C_WALL)
    return c;
  return (c & S_SIDE) ? P_WHITE : P_BLACK;
}
static inline Piece getType(Piece c) {
  return c & S_TAKEPIECE;
}
static inline bool getHorizontal(Piece c) {
  if(c == C_EMPTY || c == C_WALL)
    throw "Getting Direction of An Empty Piece";
  return c & S_HORIZONTAL;
}
static inline bool getPromoted(Piece c) {
  if(c == C_EMPTY || c == C_WALL)
    throw "Getting Promoted of An Empty Piece";
  return c & S_PROMOTED;
}
//Conversions for players and colors
namespace PlayerIO {
  std::string colorToChar(Piece c);
  std::string playerToStringShort(Player p);
  std::string playerToString(Player p);
  bool tryParsePlayer(const std::string& s, Player& pla);
  Player parsePlayer(const std::string& s);
}

//Location of a point on the board
//(x,y) is represented as (x+1) + (y+1)*(x_size+1)
typedef short Loc;
struct Move
{
    Loc x, y; Piece z; bool change;
    Move(Loc x, Loc y, Piece z = 0, bool change = false) :x(x), y(y), z(z), change(change) {}
    Move& rot();
    Move rot_();
    operator unsigned()const;
};
bool operator==(const Move& a, const Move& b);
namespace Location
{
    extern std::vector<Move> policy;
    extern int movePolicy[(BOARD_AREA + S_NUM_PIECES - 1) * BOARD_AREA * 2];
    void RotLocs(std::vector<int>& xs, std::vector<int>& ys);
    Loc getLoc(int x, int y, int x_size);
    Loc RotLoc(Loc loc, int x_size);
    void policyInit();
    size_t toPolicy(const Move& move);
    int getX(Loc loc, int x_size);
    int getY(Loc loc, int x_size);
    Loc shiftBoard(int x, int y, int x_size, int y_size);
    void getAdjacentOffsets(short adj_offsets[8], int x_size);
    bool isAdjacent(Loc loc0, Loc loc1, int x_size);
    Loc getCenterLoc(int x_size, int y_size);
    Loc getCenterLoc(const Board& b);
    bool isCentral(Loc loc, int x_size, int y_size);
    bool isNearCentral(Loc loc, int x_size, int y_size);
    int distance(Loc loc0, Loc loc1, int x_size);
    int euclideanDistanceSquared(Loc loc0, Loc loc1, int x_size);

    std::string toString(Loc loc, int x_size, int y_size);
    std::string toString(Loc loc, const Board& b);
    std::string toStringMach(Loc loc, int x_size);
    std::string toStringMach(Loc loc, const Board& b);

    bool tryOfString(const std::string& str, int x_size, int y_size, Loc& result);
    bool tryOfString(const std::string& str, const Board& b, Loc& result);
    Loc ofString(const std::string& str, int x_size, int y_size);
    Loc ofString(const std::string& str, const Board& b);

    //Same, but will parse "null" as Board::NULL_LOC
    bool tryOfStringAllowNull(const std::string& str, int x_size, int y_size, Loc& result);
    bool tryOfStringAllowNull(const std::string& str, const Board& b, Loc& result);
    Loc ofStringAllowNull(const std::string& str, int x_size, int y_size);
    Loc ofStringAllowNull(const std::string& str, const Board& b);

    std::vector<Loc> parseSequence(const std::string& str, const Board& b);
}
std::string to_string(Move mv);

//Simple structure for storing moves. Not used below, but this is a convenient place to define it.
//STRUCT_NAMED_PAIR(Loc,loc,Player,pla,Move);

//Fast lightweight board designed for playouts and simulations, where speed is essential.
//Simple ko rule only.
//Does not enforce player turn order.

struct Board
{
  //Initialization------------------------------
  //Initialize the zobrist hash.
  //MUST BE CALLED AT PROGRAM START!
  static void initHash();
  const int action_size = 4618;
  //Board parameters and Constants----------------------------------------

  static constexpr int MAX_LEN = COMPILE_MAX_BOARD_LEN;  //Maximum edge length allowed for the board
  static constexpr int DEFAULT_LEN = std::min(MAX_LEN,19); //Default edge length for board if unspecified
  static constexpr int MAX_PLAY_SIZE = MAX_LEN * MAX_LEN;  //Maximum number of playable spaces
  static constexpr int MAX_ARR_SIZE = (MAX_LEN + 1) * (MAX_LEN + 2) + 1;  // Maximum size of arrays needed

  //Location used to indicate an invalid spot on the board.
  static constexpr Loc NULL_LOC = 0;
  //Location used to indicate a pass move is desired.
  static constexpr Loc PASS_LOC = 1;

  //Zobrist Hashing------------------------------
  static bool IS_ZOBRIST_INITALIZED;
  static Hash128 ZOBRIST_SIZE_X_HASH[MAX_LEN+1];
  static Hash128 ZOBRIST_SIZE_Y_HASH[MAX_LEN+1];
  static Hash128 ZOBRIST_BOARD_HASH[MAX_ARR_SIZE][NUM_BOARD_COLORS];
  static Hash128 ZOBRIST_STAGENUM_HASH[STAGE_NUM_EACH_PLA];
  static Hash128 ZOBRIST_STAGELOC_HASH[MAX_ARR_SIZE * 2][STAGE_NUM_EACH_PLA];
  static Hash128 ZOBRIST_NEXTPLA_HASH[4];
  static Hash128 ZOBRIST_MOVENUM_HASH[MAX_MOVE_NUM];
  static Hash128 ZOBRIST_KOMA_HASH[4][MAX_KOMA_SIZE][MAX_KOMA_NUM];
  static Hash128 ZOBRIST_REPETITION_HASH[MAX_REPETITION_NUM + 1];
  static Hash128 ZOBRIST_REPLOSS_HASH[2];
  static Hash128 ZOBRIST_PLAYER_HASH[4];
  static const Hash128 ZOBRIST_GAME_IS_OVER;
  Loc KING_INIT_LOCATION[4];
  //Structs---------------------------------------

  //Constructors---------------------------------
  Board();  //Create Board of size (DEFAULT_LEN,DEFAULT_LEN)
  Board(int x, int y); //Create Board of size (x,y)
  Board(const Board& other);
  Board(const Board& other, bool temp);

  Board& operator=(const Board&) = default;

  //Functions------------------------------------

  bool isLegal(Loc loc, Player pla) const;
  //Check if this location is on the board
  bool isOnBoard(int x, int y) const;
  bool isOnBoard(Loc loc) const;
  //Is this board empty?
  bool isEmpty() const;
  //Count the number of stones on the board
  int numStonesOnBoard() const;
  int numPlaStonesOnBoard(Player pla) const;
  static int get_action_size();

  bool dummy;
  //Sets the specified stone if possible, including overwriting existing stones.
  //Resolves any captures and/or suicides that result from setting that stone, including deletions of the stone itself.
  //Returns false if location or color were out of range.
  bool setStone(Loc loc, Piece color);

  // Same, but sets multiple stones, and only requires that the final configuration contain no zero-liberty groups.
  // If it does contain a zero liberty group, fails and returns false and leaves the board in an arbitrarily changed but
  // valid state. Also returns false if any location is specified more than once.
  //bool setStones(std::vector<Move> placements);

  //Plays the specified move, assuming it is legal.
  void playMoveAssumeLegal(Move mv, Player pla);
  void playPolicy(int p);
  size_t moveToPolicy(Move mv);
  // who plays the next next move
  Player nextnextPla() const;

  // who plays the last move
  Player prevPla() const;
  
  Hash128 getSitHash(Player pla) const;
  

  //Run some basic sanity checks on the board state, throws an exception if not consistent, for testing/debugging
  void checkConsistency() const;
  //For the moment, only used in testing since it does extra consistency checks.
  //If we need a version to be used in "prod", we could make an efficient version maybe as operator==.
  bool isEqualForTesting(const Board& other) const;

  static Board parseBoard(int xSize, int ySize, const std::string& s);
  static Board parseBoard(int xSize, int ySize, const std::string& s, char lineDelimiter);
  void printBoard(std::ostream& out, const std::vector<Move>* hist)const;
  static std::string toStringSimple(const Board& board, char lineDelimiter);

  //Data--------------------------------------------

  int x_size;                  //Horizontal size of board
  int y_size;                  //Vertical size of board
  Piece colors[MAX_ARR_SIZE];  //Color of each location on the board.
  int movenum; //how many moves

  /* PointList empty_list; //List of all empty locations on board */
  Hash128 getPosHash() const {
    int repind = raw_pos_hash.hash0 < raw_pos_hash.hash1 ? 0 : 1;
    auto rephash = repind ? raw_pos_hash.hash1 : raw_pos_hash.hash0;
    int repcount = repetition[repind].count(rephash) ? repetition[repind].at(rephash) : 0;
    int willdraw = willDraw[repind].count(rephash);
    return raw_pos_hash ^ ZOBRIST_REPETITION_HASH[repcount] ^ ZOBRIST_REPLOSS_HASH[willdraw] ^
           ZOBRIST_MOVENUM_HASH[movenum];
  }
  Hash128 getPosHash(Hash128 raw_hash) const {
    int repind = raw_hash.hash0 < raw_hash.hash1 ? 0 : 1;
    auto rephash = repind ? raw_hash.hash1 : raw_hash.hash0;
    int repcount = repetition[repind].count(rephash) ? repetition[repind].at(rephash) : 0;
    int willdraw = willDraw[repind].count(rephash);
    return raw_hash ^ ZOBRIST_REPETITION_HASH[repcount] ^ ZOBRIST_REPLOSS_HASH[willdraw] ^
           ZOBRIST_MOVENUM_HASH[movenum];
  }
  // A zobrist hash of the current board position
                                                                    // (does not include ko point or player to move)
  Hash128 raw_pos_hash;
  /*
4 0 5
1   2
6 3 7
  */
  short adj_offsets[8];
  static short adj_ofx[8], adj_ofy[8];
  
  //which stage. Normally 0 = choosing piece. 1 = where to place
  //who plays the next move
  bool genLegalMove(std::vector<Move>& mvs, Player pla)const;
  std::vector<Move> genLegalMove()const;
  std::vector<size_t> genLegalPolicy()const;
  std::vector<int> get_legal_moves();
  Player nextPla;
  Loc kings[4];
  int koma[4][MAX_KOMA_SIZE];
  //一步内每一阶段的选点
  //例如：象棋类midLoc[0]是选择的棋子，midLoc[1]是落点
  //Loc midLocs[STAGE_NUM_EACH_PLA];
  int winner;
  std::unordered_map<uint64_t, int> repetition[2], first[2];
  std::unordered_set<uint64_t> willDraw[2];
  std::vector<int> checkRecord[4];
  static void printlocation(Loc loc, Board& board, string prompt);
  private:
  void init(int xS, int yS);

  friend std::ostream& operator<<(std::ostream& out, const Board& board);


  //static void monteCarloOwner(Player player, Board* board, int mc_counts[]);
};



#endif // GAME_BOARD_H_
