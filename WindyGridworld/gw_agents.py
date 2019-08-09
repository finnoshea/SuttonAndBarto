#! /usr/bin/env python3
"""
Agents for gridworld.
"""

from random import randint
from random import uniform
from random import choice
import h5py
import ast
import os

class epsilonGreedyAgent(object):
    """

    """
    def __init__(self,gridworld,epsilon=0.1,discount=1.0,name=None):
        """
        An epsilon-greedy base agent class.  The agent saves action-state value
        pairs as a nested dictionary of the form:
        {{'state0':{'action0':value,'action1':value...}},
         {'state1':{'action0':value,'action1':value...}}...}

        Parameters
        ----------
        gridworld : a gridworld object
            A pointer to a gridworld object that the agent can interact with.
        epsilon : float, optional
            A value between [0,1] for the fraction of play decisions that should
            result in a random choice of any of the possible moves.
        discount : float, optional
            How much to discount rewards from the future.  Called gamma in the
            literature by Sutton and Barto.
        name : str, optional
            A name for the agent.  If no name is given, a random 8-digit name
            will be created.  There is no guarantee that this name will be
            unique as it is randomly generated.
        """
        self.gw = gridworld
        self.epsilon = epsilon
        self.discount = discount
        if not name:
            # create a random 8-digit ID for the agent
            self.name = ''.join([str(randint(0,9)) for _ in range(8)])
        else:
            self.name = name
        self.agent = self._makeNewAgent() # action-state value pairs – the agent
        self.history = [] # list of state,action,reward values seen

    def _makeNewAgent(self):
        """
        Make a brand new agent that knows nothing.

        Returns
        -------
        An agent with knowledge of only the starting state of the girdworld.
        """
        return {self.gw.start_loc : \
                    self.enumeratePossibleActions(self.gw.start_loc)}

    def reInitAgent(self):
        """
        Reinitializes the agent to have a clear history.
        """
        self.gw.initGridWorld()
        self.history = []

    def enumeratePossibleActions(self,state=None):
        """
        Enumerates the allowed actions given a state.

        Parameters
        ----------
        state : tuple
            A location on grid world.

        Returns
        -------
        A dictionary of the allowed actions as the keys and a list [0,0] as the
        values.  The list is [Q(s,a),n] where n is the number of times Q(s,a)
        has been encountered.
        """
        # get the allowed actions
        actions = list(self.gw.validActions(state).keys())
        valid_actions = {}
        for act in actions:
            valid_actions[act] = [0.0,0]
        return valid_actions

    def takeAction(self):
        """
        Picks an action from a given state using the epsilon-greedy algorithm.

        This is the agent's policy.  It is an epsilon-greedy policy.

        Parameters
        ----------
        state : tuple
            A location on grid world.

        Returns
        -------
        The action the agents wants to take.
        """
        # check that state is in action-state dictionary
        self.checkForState(self.gw.getCurrentLoc())
        # if a uniform comes up bigger than epsilon, exploit
        if uniform(0,1) > self.epsilon:
            move = self.exploit(self.gw.getCurrentLoc())
            not_terminal, SAR = self.gw.makeMove(move)
        else: # otherwise, explore
            move = self.explore(self.gw.getCurrentLoc())
            not_terminal, SAR = self.gw.makeMove(move)
        # update history and learn
        if SAR:
            self.history.append(SAR)
            self.updateQ()
        # if you reached the terminal state
        if not not_terminal:
            self.checkForState(self.gw.getCurrentLoc())
            self.history.append((self.gw.getCurrentLoc(),0,0))
            self.updateQ() # do the last update
        return not_terminal

    def exploit(self,state):
        """
        Exploits its experience by picking an action from the list of most
        rewarding actions.  Ties are broken randomly.

        Parameters
        ----------
        state : tuple
            A location on grid world.

        Returns
        -------
        An integer representing the action to take.
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
        state : tuple
            A location on grid world.

        Returns
        -------
        An integer representing the action to take.
        """
        return choice([x for x in self.agent[state].keys()])

    def expectedStateValue(self,state):
        """
        Expected value for state.

        State values are computed as the expected value of the outcome of taking
        any action using the agent's current policy.

        Parameters
        ----------
        state : tuple of two integers
            The position on the game board to be evaluated.
        """
        try:
            vals = [x[0] for x in self.agent[state].values()]
        except KeyError:
            raise KeyError('Invalid state supplied to expectedStateValue.')
        return self.epsilon * sum(vals) / len(vals) + \
                (1 - self.epsilon) * max(vals)

    def returnGreedyPolicy(self,state):
        """
        Returns the action with the highest value for the given state.

        Parameters
        ----------
        state : tuple of two integers
            The position on the game board to be evaluated.
        """
        try:
            vals = [(k,x[0]) for k,x in self.agent[state].items()]
        except KeyError:
            raise KeyError('Invalid state supplied to returnGreedyPolicy.')
        max_key = vals[0][0]
        max_val = vals[0][1]
        for t in vals[1:]:
            if t[1] > max_val:
                max_val = t[1]
                max_key = t[0]
        return max_key

    def checkForState(self,state):
        """
        Checks the agent to make sure that it has seen state before.  If not
        the agent is updated to create the state.

        Parameters
        ----------
        state : tuple
            The position on the board to be checked for.
        """
        try:
            self.agent[state]
        except KeyError: # if not, create it in the agent
            self.agent[state] = self.enumeratePossibleActions(state)

    def updateQ(self):
        """
        Stub function for replacement by different type learning agents.
        """
        pass

    def summarizeHistory(self):
        """
        Returns summary values for the history stored by the agent.

        Returns
        -------
        A tuple containing the number of steps taken and the sum of the rewards
        seen during those steps.
        """
        sum_r = sum([x[-1] for x in self.history])
        return (len(self.history) - 1, sum_r)

    def saveAgent(self,location='agents'):
        """
        Save the agent as a string in an hdf5 file.  Agents are stored in a file
        named agent_name.hdf5.

        Parameters
        ----------
        location : string
            A string giving the location of the agents relative to the current
            working directory. Given as <current working directory>/location
        """
        filename = os.path.join(os.getcwd(),location,
                                'agent_' + self.name + '.hdf5')
        with h5py.File(filename,'w') as f:
            for attribute in self.__dict__.keys():
                if attribute == 'agent':
                    f['agent_dictionary'] = str(self.agent)
                else: # store other attributes as attributes in the HDF5
                    f.attrs[attribute] = self.__getattribute__(attribute)

    def loadAgent(self,name,location='agents'):
        """
        Load an agent from a file.

        Parameters
        ----------
        name : str
            The name of the agent to open.
        location : string
            A string giving the location of the agents relative to the current
            working directory. Given as <current working directory>/location
        """
        filename = os.path.join(os.getcwd(),location,
                                'agent_' + name + '.hdf5')
        with h5py.File(filename,'r') as f:
            # read in the agent dictionary
            self.agent = ast.literal_eval(f['agent_dictionary'][()])
            for key,value in f.attrs.items(): # grab the other attributes
                self.__setattr__(key,value)

class sarsaAgent(epsilonGreedyAgent):
    def __init__(self,gridworld,alpha=0.1,epsilon=0.1,discount=1.0,name=None):
        """
        An agent that learns using the SARSA scheme and epsilson-greedy
        behavior.

        Parameters
        ----------
        gridworld : a gridworld object
            A pointer to a gridworld object that the agent can interact with.
        alpha : float, optional
            The step size parameter for learning.
        epsilon : float, optional
            A value between [0,1] for the fraction of play decisions that should
            result in a random choice of any of the possible moves.
        discount : float, optional
            How much to discount rewards from the future.  Called gamma in the
            literature by Sutton and Barto.
        name : str, optional
            A name for the agent.  If no name is given, a random 8-digit name
            will be created.  There is no guarantee that this name will be
            unique as it is randomly generated.
        """
        super().__init__(gridworld,epsilon,discount,name)
        self.alpha = alpha # update step size

    def updateQ(self):
        """
        Perform a SARSA update of the agent. See Sutton and Barto page 130.

        Q(S,A) = Q(S,A) + alpha*[R + gamma*Q(S',A') - Q(S,A)]
        """
        if len(self.history) < 2:
            return # not enough history yet
        S = self.history[-2][0]
        A = self.history[-2][1]
        R = self.history[-2][2]
        Sp = self.history[-1][0]
        Ap = self.history[-1][1]
        self.agent[S][A][0] += self.alpha * (R + \
                                    self.discount * self.agent[Sp][Ap][0] - \
                                    self.agent[S][A][0])
        self.agent[S][A][1] += 1 # update the visit count

class QAgent(epsilonGreedyAgent):
    def __init__(self,gridworld,alpha=0.1,epsilon=0.1,discount=1.0,name=None):
        """
        An agent that learns using the Q-learning scheme and epsilson-greedy
        behavior.

        Parameters
        ----------
        gridworld : a gridworld object
            A pointer to a gridworld object that the agent can interact with.
        alpha : float, optional
            The step size parameter for learning.
        epsilon : float, optional
            A value between [0,1] for the fraction of play decisions that should
            result in a random choice of any of the possible moves.
        discount : float, optional
            How much to discount rewards from the future.  Called gamma in the
            literature by Sutton and Barto.
        name : str, optional
            A name for the agent.  If no name is given, a random 8-digit name
            will be created.  There is no guarantee that this name will be
            unique as it is randomly generated.
        """
        super().__init__(gridworld,epsilon,discount,name)
        self.alpha = alpha # update step size

    def updateQ(self):
        """
        Perform a SARSA update of the agent. See Sutton and Barto page 130.

        Q(S,A) = Q(S,A) + alpha*[R + gamma* max Q(S',a) - Q(S,A)]
        """
        if len(self.history) < 2:
            return # not enough history yet
        S = self.history[-2][0]
        A = self.history[-2][1]
        R = self.history[-2][2]
        Sp = self.history[-1][0]
        max_val = self.agent[Sp][0][0] # find the maximum Q(S',A')
        for val in self.agent[Sp].values():
            if val[0] > max_val:
                max_val = val[0]
        self.agent[S][A][0] += self.alpha * (R + \
                                    self.discount * max_val - \
                                    self.agent[S][A][0])
        self.agent[S][A][1] += 1 # update the visit count

class expectedsarsaAgent(epsilonGreedyAgent):
    def __init__(self,gridworld,alpha=0.1,epsilon=0.1,discount=1.0,name=None):
        """
        An agent that learns using the expected SARSA scheme and epsilson-greedy
        behavior.

        Parameters
        ----------
        gridworld : a gridworld object
            A pointer to a gridworld object that the agent can interact with.
        alpha : float, optional
            The step size parameter for learning.
        epsilon : float, optional
            A value between [0,1] for the fraction of play decisions that should
            result in a random choice of any of the possible moves.
        discount : float, optional
            How much to discount rewards from the future.  Called gamma in the
            literature by Sutton and Barto.
        name : str, optional
            A name for the agent.  If no name is given, a random 8-digit name
            will be created.  There is no guarantee that this name will be
            unique as it is randomly generated.
        """
        super().__init__(gridworld,epsilon,discount,name)
        self.alpha = alpha # update step size

    def updateQ(self):
        """
        Perform a SARSA update of the agent. See Sutton and Barto page 130.

        Q(S,A) = Q(S,A) + alpha*[R + gamma* max Q(S',a) - Q(S,A)]
        """
        if len(self.history) < 2:
            return # not enough history yet
        S = self.history[-2][0]
        A = self.history[-2][1]
        R = self.history[-2][2]
        Sp = self.history[-1][0]
        # find the maximum Q(S',A')
        max_val = self.agent[Sp][0][0]
        for val in self.agent[Sp].values():
            if val[0] > max_val:
                max_val = val[0]
        exp_prob = self.epsilon / len(self.agent[Sp].keys())
        expect = (1 - self.epsilon) * max_val + \
                 exp_prob * sum([x[0] for x in self.agent[Sp].values()])
        self.agent[S][A][0] += self.alpha * (R + \
                                    self.discount * expect - \
                                    self.agent[S][A][0])
        self.agent[S][A][1] += 1 # update the visit count

# some immediate testing
if __name__ == "__main__":
    from gridworld import gridWorld
    t = gridWorld()
    a = sarsaAgent(t)
