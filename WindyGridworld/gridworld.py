#! /usr/bin/env python3

"""
Game boards for the gridworld examples of chapter 6 in Sutton and Barto.
"""

class gridWorld(object):
    def __init__(self,size=(7,10),start_loc=(3,0),goal_loc=(3,7),
                 max_moves=None):
        """
        A standard GridWorld board with a start location and goal location. The
        default board looks like the one shown on page 130 of Sutton and Barto
        but without the "wind."

        See the method validMove for a list of allowed moves for this gridWorld.

        Parameters
        ----------
        size : list or tuple of two integers
            Specifies the size of the grid world in rows and columns.
        start_loc : list or tuple of two integers
            Specifies the location of the starting grid point.
        goal_loc : list or tuple of two integers
            Specifies the location of the goal grid point.
        max_moves : int
            The maximum number of allowed moves before the episode is forced to
            end.  Default is None.
        """
        self.size = tuple(size)
        self.start_loc = tuple(start_loc)
        self.goal_loc = tuple(goal_loc)
        self.max_moves = max_moves
        # the reward for all non-terminal moves is hard coded
        self.reward = -1
        # a list of allowed moves
        self.allowed_actions = self.validActions()
        # start up the grid world
        self.initGridWorld()

    def initGridWorld(self):
        """
        Initializes the grid world by initializing parameters that might vary
        from episode to episode.

        This function is intended to be used in the following manner:
        self.current, self.moves = self.initGridWorld()

        Parameters
        ----------
        None.

        Returns
        -------
        Nothing.  The attributes are modified in place.
        """
        self.current_loc = self.start_loc
        self.moves = 0

    def validActions(self,state=None):
        """
        A enumeration of the moves allowed by the grid world at the given state.
        These are the actions the agent can take at any space.

        This is the function you modify to change the allowed actions.

        For a basic grid world, the actions are the same for every state.

        Parameters
        ----------
        state : int, optional
            The state for which you want to return the valid actions.

        Returns
        -------
        Returns a dictionary mapping integers to functions that produce
        movement in the grid world.
        """

        def move_up():
            return (-1,0)
        def move_down():
            return (1,0)
        def move_left():
            return (0,-1)
        def move_right():
            return (0,1)

        allowed_actions = {0: move_up,
                           1: move_down,
                           2: move_left,
                           3: move_right
                          }

        return allowed_actions

    def makeMove(self,move):
        """
        Function for an agent to change its position on the board.

        Parameters
        ----------
        move : an integer representing a move on the board

        Returns
        -------
        A boolean indicating if the episode is still live (i.e. the agent has
        not yet reached a goal state or run out of moves) and a tuple made up of
        (state,move,reward).  If this function returns True, None, it means that
        an invalid move was requested.

        """
        old_loc = self.current_loc
        new_loc = self.validMove(move)
        if new_loc: # the move was valid
            self.current_loc = new_loc
            self.moves += 1
        else:
            return True, None # the move was invalid

        if self.current_loc == self.goal_loc: # made it to the goal state
            return False, (old_loc,move,0)
        if self.max_moves: # make sure max_moves is not None
            if self.moves == self.max_moves: # ran out of moves
                return False, (old_loc,move,self.reward)
        return True, (old_loc,move,self.reward)

    def validMove(self,move):
        """
        A catch-all function for evaluating move validity.  This function is
        for determining whether a move is legal by the gridWorld rules and _NOT_
        for punishing bad moves or good moves.

        Define valid actions in the validActions method.

        For the basic gridWorld only up, down, left and right are allowed.  So
        if an agent requested to move two spaces to the upper right, this
        function will call it invalid.

        Parameters
        ----------
        move : int
            An integer representing the move.

        Returns
        -------
        If the move is valid, return the new location.  Otherwise, return None.
        """
        try:
            delta = self.allowed_actions[move].__call__()
        except KeyError:
            return None
        new_loc = (self.current_loc[0] + delta[0],
                   self.current_loc[1] + delta[1])

        if self._bounds_check(new_loc):
            return new_loc
        else: # if the new location fails a bounds check, return current_loc
            return self.current_loc

    def getCurrentLoc(self):
        return self.current_loc

    def _bounds_check(self,loc):
        """ checks to make sure a location is in bounds """
        if -1 in loc:
            return False
        elif loc[0] >= self.size[0]:
            return False
        elif loc[1] >= self.size[1]:
            return False
        return True

    def print_board(self):
        """ prints the board using ASCII characters """
        board = [['0' for i in range(self.size[1] + 2)]]
        for _ in range(self.size[0]):
                board.append(['0']+[" " for _ in range(self.size[1])] + ['0'])
        board.append(['0' for _ in range(self.size[1] + 2)])
        board[self.start_loc[0]+1][self.start_loc[1]+1] = 'S'
        board[self.goal_loc[0]+1][self.goal_loc[1]+1] = 'G'
        board[self.current_loc[0]+1][self.current_loc[1]+1] = 'X'
        for row in board:
            print(''.join(row))

class kingWorld(gridWorld):
    """

    """
    def __init__(self,size=(7,10),start_loc=(3,0),goal_loc=(3,7),
                 max_moves=None):
        """
        A GridWorld board with a start location, goal location and king moves. The
        default board looks like the one shown on page 130 of Sutton and Barto
        but without the "wind."

        See the method validMove for a list of allowed moves for this gridWorld.

        Parameters
        ----------
        size : list or tuple of two integers
            Specifies the size of the grid world in rows and columns.
        start_loc : list or tuple of two integers
            Specifies the location of the starting grid point.
        goal_loc : list or tuple of two integers
            Specifies the location of the goal grid point.
        max_moves : int
            The maximum number of allowed moves before the episode is forced to
            end.  Default is None.
        """
        super().__init__(size,start_loc,goal_loc,max_moves)

    def validActions(self,state=None):
        """
        A enumeration of the moves allowed by the grid world at the given state.
        These are the actions the agent can take at any space.

        This is the function you modify to change the allowed actions.

        For a basic grid world, the actions are the same for every state.

        Parameters
        ----------
        state : int, optional
            The state for which you want to return the valid actions.

        Returns
        -------
        Returns a dictionary mapping integers to functions that produce
        movement in the grid world.
        """

        def move_up():
            return (-1,0)
        def move_down():
            return (1,0)
        def move_left():
            return (0,-1)
        def move_right():
            return (0,1)
        def move_ul():
            return (-1,-1)
        def move_ur():
            return (-1,1)
        def move_ll():
            return (1,-1)
        def move_lr():
            return (1,1)

        allowed_actions = {0: move_up,
                           1: move_down,
                           2: move_left,
                           3: move_right,
                           4: move_ul,
                           5: move_ur,
                           6: move_ll,
                           7: move_lr
                          }

        return allowed_actions
