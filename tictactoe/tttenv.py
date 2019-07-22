#! /usr/bin/env python3
"""
A tic-tac-toe environment.
"""

from random import choice
from tttboard import TicTacToe
from tttagent import EpsGreedyTTTAgent
from tttagent import EpsGreedyDecayTTTAgent
from tttagent import EpsGreedyDecayQAgent
import os

# create a single player to play itself
player_one = EpsGreedyTTTAgent()
player_two = player_one

# create the tic-tac-toe game
game = TicTacToe()

class TTTenv(object):
    """

    """

    def __init__(self,player_one=None,player_two=None,
                      game=None,random_board=False):
        """
        An environment for having two players play tic-tac-toe.

        Parameters
        ----------

        player_one : a tic-tac-toe player agent
            An example is EpsGreedyTTTAgent from tttagent.py.
        player_two : a tic-tac-toe player agent
            Similar to player_one.  If this agent is set to None, player_one
            will play itself.
        game : str
            A 9-digit string of 0s,1s, and 2s to make a complete tic-tac-toe
            board.  The default is an empty board.
        random_board : boolean
            True if you want to generate a random board.  WARNING: random boards
            might already have a winner.

        """
        if not player_one:
            self.player_one = EpsGreedyDecayQAgent()
        else:
            self.player_one = player_one
        if not player_two:
            self.player_two = self.player_one
        else:
            self.player_two = player_two
        if not game:
            self.game = TicTacToe(random_board=random_board)
        else:
            self.game = TicTacToe(board=game)

    def play_game(self):
        """
        Uses player_one and player_two to play a game of tic-tac-toe.

        Returns
        -------
        Two lists of [state,action,reward].  One for each player.
        """
        p_one_hist = []
        p_two_hist = []
        win_value = 1.0
        los_value = -1.0
        dra_value = 0.0
        # 0: on-going draw, -1: final draw, 1 or 2: the winner
        while self.game.winner == 0:
            temp_list = [self.game.board]
            if self.game.plays % 2 == 0: # player_one's turn
                action = self.player_one.choose_move(self.game.board)
                temp_list += [action]
                self.game.make_play(action,1) # this updates self.game.plays
                if self.game.winner == 1:
                    temp_list += [win_value] # player one wins
                    if p_two_hist: # make sure two has played
                        p_two_hist[-1][-1] = los_value # player two loses
                elif self.game.winner == -1: # final draw, only on player 1 turn
                    temp_list += [dra_value] # player one draws
                    if p_two_hist: # make sure two has played
                        p_two_hist[-1][-1] = dra_value # player two draws, too
                else:
                    temp_list += [0]
                p_one_hist += [temp_list]
            elif self.game.plays % 2 == 1: # player_two's turn
                action = self.player_two.choose_move(self.game.board)
                temp_list += [action]
                self.game.make_play(action,2) # this updates self.game.plays
                if self.game.winner == 2:
                    temp_list += [win_value] # player two wins
                    if p_one_hist: # make sure one has played
                        p_one_hist[-1][-1] = los_value # player one loses
                else:
                    temp_list += [0]
                p_two_hist += [temp_list]
        return p_one_hist, p_two_hist

    def play_game_and_learn(self):
        """
        Plays a game of tic-tac-toe between player_one and player_two and then
        uses their internal learning mechanisms to update action-values.
        """
        p_one_hist, p_two_hist = self.play_game()
        self.player_one.learn_from_game(p_one_hist)
        self.player_two.learn_from_game(p_two_hist)

    def pl_n_games(self,n_games=10000):
        """
        Play and learn for n games.

        Parameters
        ----------
        n_games : int
            The number of games (episodes) to play.
        """
        for idx in range(n_games):
            self.game.reset_board()
            self.play_game_and_learn()
        self.player_one.save_agent()
        if self.player_two != self.player_one:
            self.player_two.save_agent()

    def print_game_board(self):
        """
        Prints the game board in a more human friendly format.
        """
        def char(a):
            if a == '1':
                return 'X'
            elif a == '2':
                return 'O'
            else:
                return ' '
        board = self.game.board
        print(char(board[0]) + '|' + char(board[1]) + '|' + char(board[2]))
        print('------')
        print(char(board[3]) + '|' + char(board[4]) + '|' + char(board[5]))
        print('------')
        print(char(board[6]) + '|' + char(board[7]) + '|' + char(board[8]))

    def play_human(self,player=1):
        """
        Allows a human to play against the computer as either player 1 or 2.

        Parameters
        ----------
        player : int
            Either 1 for player 1 or 2 for player 2.

        """
        self.game.reset_board() # reset the game board

        # set up the computer player to be greedy
        if player == 1:
            players = [None,self.player_two]
            self.player_two.play_greedy()
        elif player == 2:
            players = [self.player_one,None]
            self.player_one.play_greedy()
        else:
            print('Invalid player choice.  Choose either 1 or 2.')
            return None

        # play the game here:
        # 0: on-going draw, -1: final draw, 1 or 2: the winner
        while self.game.winner == 0:
            plr = players[self.game.plays % 2]
            marker = self.game.plays % 2 + 1
            # someone gets to make a play
            if not plr: # if not None = human's turn
                print('Game board:')
                self.print_game_board()
                m = int(input('Choose your square: '))
                self.game.make_play(m,marker)
            else: # computer's turn
                self.game.make_play(plr.choose_move(self.game.board),marker)

        # who is the winner?
        if self.game.winner == player:
            print('You won as player {:d}!'.format(player))
        elif self.game.winner == -1:
            print('Draw with the computer!')
        else:
            print('You lose as player {:d}!'.format(player))
        self.print_game_board()

        # set the computer player back to epsilon-greedy
        if player == 1:
            self.player_two.play_to_learn()
        else:
            self.player_one.play_to_learn()

class TTTournament(object):
    """

    """
    def __init__(self,num_agents=10,
                      agent_loc='agents',
                      base_agent=EpsGreedyDecayTTTAgent,
                      agents=[],
                      pick_existing_agents=True):
        """

        """
        self.num_agents = num_agents
        self.agent_loc = agent_loc
        self.base_agent = base_agent
        self.agents = self.find_agents(self.num_agents,
                                       agents,pick_existing_agents)

    def find_agents(self,num_agents=10,agents=[],
                         pick_existing_agents=True):
        """
        Loads agents from the the agent_loc directory.  If there aren't enough
        agents, create the remainder.

        Parameters
        ----------
        num_agents : int
            The number of agents to load or create.
        agents : list of agent names (str)
            A list of agent names to load.  It need not match num_agents in
            length.
        pick_existing_agents : boolean
            If True (default) existing agents are loaded in to memory after the
            agents given in the agents parameter.  May result in loading the
            same agent twice if it was listed in the agents parameter.

        Returns
        -------
        A list of the agents.
        """
        players = [] # the list of players
        agents = []
        for name in agents: # if agents is a non-empty list
            t = self.base_agent()
            t.load_agent(name=short_name,location=self.agent_loc)
            players += [t]
            num_agents -= 1

        def shorten_name(name):
            """ converts filenames in to agent numbers or None """
            s = name.split('.')
            if len(s) > 1: # might be an agent file
                if s[1] == 'hdf5': # could be an agent file
                    s = s[0].split('_')
                    if len(s) > 1: # probably an agent file
                        return s[1]
            return None

        if pick_existing_agents:
            existing_agents = os.listdir(
                                os.path.join(os.getcwd(),self.agent_loc))
            for name in existing_agents:
                short_name = shorten_name(name)
                if short_name: # bad files return short_name = None
                    t = self.base_agent()
                    t.load_agent(name=short_name,location=self.agent_loc)
                    players += [t]
                    num_agents -= 1

        while num_agents > 0:
            players += [self.base_agent()]
            num_agents -= 1
        return players

    def print_players(self):
        """
        Prints the agents in the tournament in human-readble form.
        """
        print('********** Players **********')
        for idx,agent in enumerate(self.agents):
            print('Player #{:d}: {:s}'.format(idx,agent.name))
        print('*****************************')

    def play_games(self,player_one=None,player_two=None,n_games=10000):
        """
        Has player_one and player_two play n_games against each other and then
        saves the players.
        """
        env = TTTenv(player_one=player_one,player_two=player_two,
                     game=None,random_board=False)
        # play the games and learn from them
        env.pl_n_games(n_games=n_games)

    def train_players(self,n_matches=10,n_games=10000):
        """
        Runs a FFA tournament with random players playing each other or even
        themselves.

        Parameters
        ----------
        n_matches : int
            The number of matches to schedule.
        n_games : int
            The number of games to play per match.
        """
        for match in range(n_matches):
            pone = choice(self.agents)
            ptwo = choice(self.agents)
            print('Match {:d}: playing {:d} games between {:s} and {:s}'\
                        .format(match,n_games,pone.name,ptwo.name))
            self.play_games(player_one=pone,
                            player_two=ptwo,
                            n_games=n_games)

    def rank_players(self):
        """ ranks the players in human-readable form. """
        ranking = []
        for num,agent in enumerate(self.agents):
            m = agent.get_matches()
            if m != 0:
                ranking += [[agent.get_wins() / m,agent.get_loses() / m,
                             num,agent.name]]
            else:
                ranking += [[0.0,0.0,num,agent.name]]
        # bubble sort the ranking by w / m
        for ida in range(len(ranking) - 1):
            for idb in range(len(ranking) - ida - 1):
                if ranking[idb][0] < ranking[idb + 1][0]:
                    # swap them
                    temp = ranking[idb]
                    ranking[idb] = ranking[idb + 1]
                    ranking[idb + 1] = temp
        print('********** Rankings **********')
        for idx,rank in enumerate(ranking):
            print('{:d} :: Player #{:d}: {:s}'\
                    .format(idx,rank[2],rank[3]),
                  ' Win rate: {:4.3f} / Lose rate: {:4.3f}'\
                        .format(rank[0],rank[1]))
        print('*****************************')
