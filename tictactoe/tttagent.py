#! /usr/bin/env python3
"""
A player agent for tic-tac-toe using reinforcement learning.
"""

from random import uniform
from random import choice
from random import randint
import h5py
import ast
import os

class TTTAgent(object):
    """

    """

    def __init__(self,epsilon=0.1,discount=1.0,name=None):
        """
        Creates a tic-tac-toe player agent.  The state space
        that the agent has experienced is created lazily.

        This agent is a purely on-policy agent that uses a Monte Carlo reward
        system.

        Parameters
        ----------
        epsilon : float
            A value between [0,1] for the fraction of play decisions that should
            result in a random choice of any of the possible moves.
        discount : float
            How much to discount rewards from the future.  Called gamma in the
            literature by Sutton and Barto.  Applies only to positive rewards.
            A loss is a loss.  This parameter should encourage aggressive play.
        name : str
            A name for the agent.  If no name is given, a random 8-digit name
            will be created.  There is no guarantee that this name will be
            unique as it is randomly generated.
        """
        self.epsilon = epsilon
        self.agent = self.make_new_agent()
        self.gamma = discount
        self.performance = [0,0,0] # win, lose, draw
        self._greedy = False # whether to always choose the most valuable action
        if not name: # no name given
            # create a random 8-digit ID for the agent
            self.name = ''.join([str(randint(0,9)) for _ in range(8)])
        else:
            self.name = name

    def make_new_agent(self):
        """
        Creates a new agent that has only seen the empty board state of a game
        of tic-tac-toe.

        Returns
        -------
        The agent in the form of a multiple level dictionary:
            {s1: {a1: [0,0], a2: [0,0] ... } ... }
        Where s1 is a state label and a1 is an action label.
        """
        return {'000000000':self.enumerate_possible_actions('000000000')}

    def enumerate_possible_actions(self,state):
        """
        Enumerates the allowed actions given a state.

        Parameters
        ----------
        state : str
            A string of 9 digits [0,2] containing a tic-tac-toe board \
            configuration. Example: '011020000'

        Returns
        -------
        A dictionary of the allowed actions as the keys and a list [0,0] as the
        values.  The list is [Q(s,a),n] where n is the number of times Q(s,a)
        has been encountered.
        """
        actions = {}
        for idx,square in enumerate(state):
            if square == '0': # an empty square
                actions[idx] = [0.0,0]
        return actions

    def check_for_state(self,state):
        """
        Checks the agent to make sure that it has seen state before.  If not
        the agent is updated to create the state.

        Parameters
        ----------
        state : str
            A string of 9 digits [0,2] containing a tic-tac-toe board \
            configuration. Example: '011020000'
        """
        try:
            self.agent[state]
        except KeyError: # if not, create it in the agent
            self.agent[state] = self.enumerate_possible_actions(state)

    def update_action_value(self,state,action,reward):
        """
        Updates the Q(s,a) value for the given state and action by using the
        reward and the incremental formula:
        Q(n+1) = Q(n) + (R(n) + Q(n)) / n
        Also increase n by 1.

        Parameters
        ----------
        state : str
            A string of 9 digits [0,2] containing a tic-tac-toe board \
            configuration. Example: '011020000'
        action : int
            The action taken to recieve the reward.
        reward : float
            The reward received for taking the action from the state.

        Returns
        -------
        Nothing.  The agent is modified in place.
        """
        # increment the number of times this state, action pair has been seen
        try:
            self.agent[state][action]
        except KeyError:
            print('Error when attempting to access:')
            print('State: {:s}, action: {:d}'.format(state,action))
        self.agent[state][action][1] += 1
        self.agent[state][action][0] += (reward - self.agent[state][action][0])\
                                        / self.agent[state][action][1]

    def learn_from_game(self,history):
        """
        Uses the Monte Carlo update on page 99 of Sutton and Barto to update
        the various action-values, Q(s,a), encountered during a game.

        Parameters
        ----------
        history : list of lists
            A list of triplets of the form [state,action,reward] to be used in
            the Monte Carlo update of the Q(s,a).

        Returns
        -------
        Nothing, the agent is updated in place.

        """
        G = 0
        for triplet in reversed(history):
            G += triplet[2] # increase G by reward.
            self.update_action_value(triplet[0],triplet[1],G)
        # count wins or loses
        if G > 0: # G will be 1 if the agent won
            self.performance[0] += 1
            G *= self.gamma # discount the reward for earlier states
        elif G < 0: # lose
            self.performance[1] += 1
        else: # draw
            self.performance[2] += 1

    def print_state(self,state):
        """
        Prints the actions with their corresponding action-values and the
        number of times the action has been encountered.

        Parameters
        ----------
        state : str
            A string of 9 digits [0,2] containing a tic-tac-toe board \
            configuration. Example: '011020000'

        Returns
        -------
        The described print out.
        """
        the_state = self.agent[state]
        digits = len(str(max([x[1] for x in the_state.values()])))
        floats = max([len('{:4.3f}'.format(x[0])) for x in the_state.values()])
        print('Agent {:s}'.format(self.name))
        print('************************************')
        print('Contents of state {:s}'.format(state))
        print('Visits: {:d}'.format(sum([x[1] for x in the_state.values()])))
        for k,v in the_state.items():
            print('Action {:d}, seen {:{digits}d}'.format(k,v[1],digits=digits),
                  'times with action-value: {:{floats}.3f}'\
                        .format(v[0],floats=floats))
        print('************************************')

    def get_wins(self):
        """ Returns the number of wins the agent has. """
        return self.performance[0]

    def get_loses(self):
        """ Returns the number of loses the agent has. """
        return self.performance[1]

    def get_draws(self):
        """ Returns the number of draws the agent has. """
        return self.performance[2]

    def get_matches(self):
        """ Returns the number of matches the agent has played. """
        return sum(self.performance)

    def save_agent(self,location='agents'):
        """
        Save the agent as a string in an hdf5 file.  Agents are stored in a file
        named agent_name.hdf5.

        Parameters
        ----------
        location : string
            A string giving the location of the agents relative to the current
            working directory. Given as <current working directory>/location

        The file has the following structure:
        f.attrs['agent_name'] : the agent's name
        f.attrs['epislon']    : the epsilon setting for the agent
        f['agent_dictionary'] : the agent's action-value dictionary
        """
        filename = os.path.join(os.getcwd(),location,
                                'agent_' + self.name + '.hdf5')
        with h5py.File(filename,'w') as f:
            for attribute in self.__dict__.keys():
                if attribute == 'agent':
                    f['agent_dictionary'] = str(self.agent)
                else: # store other attributes as attributes in the HDF5
                    f.attrs[attribute] = self.__getattribute__(attribute)

    def load_agent(self,name,location='agents'):
        """
        Load an agent from a file.

        Parameters
        ----------
        name : str
            The name of the agent to open.
        location : string
            A string giving the location of the agents relative to the current
            working directory. Given as <current working directory>/location

        The file has the following structure:
        f.attrs['agent_name'] : the agent's name
        f.attrs['epislon']    : the epsilon setting for the agent
        f['agent_dictionary'] : the agent's action-value dictionary
        """
        filename = os.path.join(os.getcwd(),location,
                                'agent_' + name + '.hdf5')
        with h5py.File(filename,'r') as f:
            # read in the agent dictionary
            self.agent = ast.literal_eval(f['agent_dictionary'][()])
            for key,value in f.attrs.items(): # grab the other attributes
                self.__setattr__(key,value)


class EpsGreedyTTTAgent(TTTAgent):
    """

    """

    def __init__(self,epsilon=0.1):
        """
        Creates an epsilon-greedy tic-tac-toe player agent.  The state space
        that the agent has experienced is loaded lazily.

        """
        super().__init__(epsilon=epsilon)

    def play_greedy(self):
        """
        After this method is called, the agent will always choose the most
        valuable action.  This is usually bad for learning.
        """
        self._greedy = True

    def play_to_learn(self):
        """
        After this method is called, the agent will play as an epsilon-greedy
        agent according to its internal settings that were set at instantiation.
        """
        self._greedy = False

    def choose_move(self,state):
        """
        Picks an action from a given state using the epsilon-greedy algorithm.

        This is the agent's policy.  It is an epsilon-greedy policy.

        Parameters
        ----------
        state : str
            A string of 9 digits [0,2] containing a tic-tac-toe board \
            configuration. Example: '011020000'

        Returns
        -------
        The action the agents wants to take.
        """
        self.check_for_state(state)
        # if a randint comes up bigger than epsilon or the agent is set greedy
        if uniform(0,1) > self.epsilon or self._greedy:
            return self.exploit(state)
        else:
            return self.explore(state)

    def exploit(self,state):
        """
        Exploits its experience by picking an action from the list of most
        rewarding actions.  Ties are broken randomly.

        Parameters
        ----------
        state : str
            A string of 9 digits [0,2] containing a tic-tac-toe board \
            configuration. Example: '011020000'

        Returns
        -------
        An integer representing the square to fill in, i.e. the action to take.
        """
        # start with an arbitrary q_max and a_max
        a_max = [choice(list(self.agent[state].keys()))]
        q_max = self.agent[state][a_max[0]][0]
        for key,value in self.agent[state].items():
            if value[0] > q_max:
                q_max = value[0]
                a_max = [key]
            elif value[0] == q_max:
                a_max += [key]
        return choice(a_max)

    def explore(self,state):
        """
        Explores the state space by randomly choosing an action to take.

        Parameters
        ----------
        state : str
            A string of 9 digits [0,2] containing a tic-tac-toe board \
            configuration. Example: '011020000'

        Returns
        -------
        An integer representing the square to fill in, i.e. the action to take.
        """
        return choice([x for x in self.agent[state].keys()])

class EpsGreedyDecayTTTAgent(EpsGreedyTTTAgent):
    """

    """

    def __init__(self,init_epsilon=1.0,
                      target_epsilon=0.1,
                      decay_rate_epsilon=1e-8):
        """
        An epsilon greedy agent in which epsilon is allowed to decay.

        Parameters
        ----------
        init_epsilon : float
            Initial epsilon.  0<=init_epsilon<=1
        target_epsilon : float
            Minimum value of epsilon to be reached after an infinite number of
            training epsiodes. 0<=target_epsilon<=1
        decay_rate_epsilon : float
            Relative decay rate for epsilon per episode.  See method
            update_epsilon for details. 0<=decay_rate_epsilon<=1

        """
        self.init_epsilon = init_epsilon
        self.target_epsilon = target_epsilon
        self.decay_rate_epsilon = decay_rate_epsilon
        super().__init__(epsilon=init_epsilon)

    def update_epsilon(self):
        """
        Updates epsilon by applying the following rule

        de/ds = -delta * (e - ef)

        e = ef + (e0 - ef) * exp(-delta * s)
        e = current epsilon value
        ef = target_epsilon
        e0 = init_epsilon
        delta = decay_rate_epsilon
        s = episode number

        All of these parameters are taken from the class instance.
        """
        de = -1 * self.decay_rate_epsilon * (self.epsilon - self.target_epsilon)
        return self.epsilon + de

    def learn_from_game(self,history):
        """
        Uses the Monte Carlo update on page 99 of Sutton and Barto to update
        the various action-values, Q(s,a), encountered during a game.

        Parameters
        ----------
        history : list of lists
            A list of triplets of the form [state,action,reward] to be used in
            the Monte Carlo update of the Q(s,a).

        Returns
        -------
        Nothing, the agent is updated in place.

        """
        super().learn_from_game(history) # learn from the game as epsilon-greedy
        self.epsilon = self.update_epsilon() # update epsilon

class EpsGreedyDecayQAgent(EpsGreedyTTTAgent):
    """

    """

    def __init__(self,learning_rate=0.5,
                      init_epsilon=1.0,
                      target_epsilon=0.1,
                      decay_rate_epsilon=1e-6):
        """
        An epsilon greedy agent in which epsilon is allowed to decay that uses
        Q-learning.  This agent is a purely on-policy agent.

        Parameters
        ----------
        init_epsilon : float
            Initial epsilon.  0<=init_epsilon<=1
        target_epsilon : float
            Minimum value of epsilon to be reached after an infinite number of
            training epsiodes. 0<=target_epsilon<=1
        decay_rate_epsilon : float
            Relative decay rate for epsilon per episode.  See method
            update_epsilon for details. 0<=decay_rate_epsilon<=1

        """
        self.learning_rate = learning_rate
        self.init_epsilon = init_epsilon
        self.target_epsilon = target_epsilon
        self.decay_rate_epsilon = decay_rate_epsilon
        super().__init__(epsilon=init_epsilon)

    def update_epsilon(self):
        """
        Updates epsilon by applying the following rule

        de/ds = -delta * (e - ef)

        e = ef + (e0 - ef) * exp(-delta * s)
        e = current epsilon value
        ef = target_epsilon
        e0 = init_epsilon
        delta = decay_rate_epsilon
        s = episode number

        All of these parameters are taken from the class instance.
        """
        de = -1 * self.decay_rate_epsilon * (self.epsilon - self.target_epsilon)
        return self.epsilon + de

    def learn_from_game(self,history):
        """
        Uses the Monte Carlo update on page 99 of Sutton and Barto to update
        the various action-values, Q(s,a), encountered during a game.

        Parameters
        ----------
        history : list of lists
            A list of triplets of the form [state,action,reward] to be used in
            the Monte Carlo update of the Q(s,a).

        Returns
        -------
        Nothing, the agent is updated in place.

        """

        next_state = None
        next_action = None
        G = 0
        for m,triplet in enumerate(reversed(history)):
            G += triplet[2] # may want to use this in place of triplet[2]
            self.update_action_value(triplet[0],triplet[1],
                                     triplet[2],next_state,next_action)
            next_state = triplet[0]
            next_action = triplet[1]
        # count wins or loses
        if G > 0: # G will be 1 if the agent won
            self.performance[0] += 1
        elif G < 0: # lose
            self.performance[1] += 1
        else: # draw
            self.performance[2] += 1
        self.epsilon = self.update_epsilon() # update epsilon

    def update_action_value(self,state,action,reward,next_state,next_action):
        """
        Updates the Q(s,a) value for the given state and action using
        TD(0) update.  It still waits until the end of an epside, so it can
        use a similar learn_from_game method that the Monte Carlo learner does.
        This will slow learning a little bit.

        Parameters
        ----------
        state : str
            A string of 9 digits [0,2] containing a tic-tac-toe board \
            configuration. Example: '011020000'
        action : int
            The action taken to recieve the reward.
        reward : float
            The reward received for taking the action from the state.
        next_state : str
            The next state that the action will lead to.
        next_action : int
            The action that the agent will take in next_state.

        Returns
        -------
        Nothing.  The agent is modified in place.
        """
        try:
            self.agent[state][action] # check the state in the history
        except KeyError:
            print('Error when attempting to access:')
            print('State: {:s}, action: {:d}'.format(state,action))
        if next_state:
            q_new = self.agent[next_state][next_action][0]
        else:
            q_new = 0 # if action led to a terminal state, q_new = 0, by def
        self.agent[state][action][1] += 1 # update the # of times seen
        a = self.learning_rate
        d = self.gamma
        q = self.agent[state][action][0]
        self.agent[state][action][0] = (1 - a) * q + a * (reward + d * q_new)
