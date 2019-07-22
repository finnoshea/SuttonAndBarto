#! /usr/bin/env python3
"""
A tic-tac-toe game class.
"""

import random

class TicTacToe(object):
    """
    """

    def __init__(self,board='000000000',random_board=False):
        """
        A tic-tac-toe board stored as a single string.  The board is read from
        the top left to the bottom right.  Numbering:

        0|1|2
        -----
        3|4|5
        -----
        6|7|8

        0 -> empty square
        1 -> square taken by X
        2 -> square taken by O

        Parameters
        ----------
        board : str
            A 9-digit string of 0s,1s, and 2s to make a complete tic-tac-toe
            board.  The default is an empty board.
        random_board : bool
            True if you want to generate a random board.  WARNING: random boards
            might already have a winner.
        """
        self.random_board = random_board
        if self.random_board:
            self.turn, self.board = self.create_random_board()
            self._backup_board = self.board
            self.winner = self.declare_winner()
            self.plays = sum([1 if x != '0' else 0 for x in self.board])
        else:
            self.board = board
            self._backup_board = board
            self.turn = 1 # the player who gets to chose a square
            self.winner = 0 # no one has won yet
            self.plays = 0 # the number of plays made

    def create_random_board(self):
        """
        Creates a random tic-tac-toe board.

        Returns
        -------
        The player whose turn it is.
        The board layout.
        """
        board = [0,0,0,0,0,0,0,0,0]
        # number of Os
        ohhs = random.randint(0,4)
        # number of Xs - maybe one more than the Os
        if random.uniform(0,1) >= 0.5:
            exxs = ohhs + 1
            players_turn = 2 # X has played once more than O, it is O's turn
        else:
            exxs = ohhs
            players_turn = 1 # X has played as many time as O, it is X's turn
        plays = [0,1,2,3,4,5,6,7,8]
        random.shuffle(plays) # reorder the plays
        for idx in range(exxs + ohhs):
            if idx % 2 == 0: # X makes the even moves
                board[plays[idx]] = 1
            else: # O makes the odd moves
                board[plays[idx]] = 2
        return players_turn, self._convert_board(board)

    def make_play(self,position,player):
        """
        Puts a player's marker in the specified position.

        Parameters
        ----------
        position : int
            The position in which to put the player's marker.
        player : int
            The player's designation: 1 -> X, 2-> O

        Returns
        -------
            Nothing, the board is modified in place.
        """
        if self.board[position] == '0':
            self.board = self.board[0:position] + str(player) + \
                         self.board[position+1:]
            self.plays += 1 # add to the plays count
            self.turn = self.plays % 2 + 1
            self.winner = self.declare_winner()
        else:
            print('Invalid move. Position {:d} is already taken.'\
                  .format(position))
            self.board

    def _convert_board(self,board):
        """
        Converts the given tic-tac-toe board from a list to a string or vice
        versa.

        Parameters
        ----------
        board : str or list
            The tic-tac-toe board in either string or list form.
        """
        if isinstance(board,str):
            return [x for x in map(int,board)]
        elif isinstance(board,list):
            return ''.join(map(str,board))
        else:
            raise TypeError('Input to convert_board must be either string \
                             or list.')
            return None

    def print_board(self):
        """
        Prints the current tic-tac-toe board to standard out.
        """
        print(self.board[0:3])
        print(self.board[3:6])
        print(self.board[6:])

    def declare_winner(self,board=None):
        """
        Examines the board and declares if there is a winner.

        Returns
        -------
            An integer number representing the winner (1,2), an incomplete
            board (0) or a draw (-1).
        """
        # define some local helper functions
        def row_check(board):
            if board[0] == board[1] and board[1] == board[2] \
                                    and board[0] != 0:
                return int(board[0])
            elif board[3] == board[4] and board[4] == board[5] \
                                      and board[3] != 0:
                return int(board[3])
            elif board[6] == board[7] and board[7] == board[8] \
                                      and board[6] != 0:
                return int(board[6])
            else:
                return 0
        def col_check(board):
            if board[0] == board[3] and board[3] == board[6] \
                                    and board[0] != 0:
                return int(board[0])
            elif board[1] == board[4] and board[4] == board[7] \
                                      and board[1] != 0:
                return int(board[1])
            elif board[2] == board[5] and board[5] == board[8] \
                                      and board[2] != 0:
                return int(board[2])
            else:
                return 0
        def dia_check(board):
            if board[0] == board[4] and board[4] == board[8] \
                                    and board[0] != 0:
                return int(board[0])
            elif board[2] == board[4] and board[4] == board[6] \
                                      and board[2] != 0:
                return int(board[2])
            else:
                return 0
        if not board: # no board given
            board = self.board
        # is the board full?
        if not all(self._convert_board(board)): # if there are any 0s
            full = False
        else: # there are no zeros, the board is full
            full = True
        # check for a winner
        if row_check(board):
            return row_check(board)
        elif col_check(board):
            return col_check(board)
        elif dia_check(board):
            return dia_check(board)
        else: # no winner yet
            if full:
                return -1 # draw game
            else:
                return 0 # incomplete game

    def reset_board(self):
        """
        Resets the board to the condition it was in when the instance was
        created.
        """
        if self.random_board:
            self.turn, self.board = self.create_random_board()
            self._backup_board = self.board
            self.winner = self.declare_winner()
            self.plays = sum([1 if x != '0' else 0 for x in self.board])
        else:
            self.board = self._backup_board
            self.plays = sum([1 if x != '0' else 0 for x in self.board])
            self.winner = 0
