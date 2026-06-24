/*
 * gamelogic.h
 * Logics of game rules
 * Some other game logics are in board.h/cpp
 * 
 * Gomoku as a representive
 */
#pragma once
#ifndef GAME_GAMELOGIC_H_
#define GAME_GAMELOGIC_H_

#include "board.h"
#include <vector>
#include <iostream>
#include <functional>
using namespace std;
/*
* Other game logics:
* Board::
*/

namespace GameLogic {
  using std::vector;

  //C_EMPTY = draw, C_WALL = not finished 
  Piece checkWinnerAfterPlayed(const Board& board, Player pla, Loc loc);
  bool checkLegal(const Board& board, Loc loc, Piece piece, Loc locd, Player pla);
  bool getAttackingPiece(const Board& board, Loc loc, Player pla, vector<Loc>& res);
  bool inPromotionArea(const Board& board, Piece piece, Loc loc, int x_size, int y_size);
  bool inKnightMustPromoteArea(const Board& board, Piece piece, Loc loc);
  bool getMove(const Board& board, vector<Loc>& res, Loc loc, Piece piece);
  bool locAttacked(const Board& board, Loc loc, Player pla);
  //some results calculated before calculating NN
  //part of NN input, and then change policy/value according to this

}




#endif // GAME_RULELOGIC_H_
