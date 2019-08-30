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
    def __init__(self,gridworld,alpha=0.5,epsilon=0.1,discount=1.0,name=None):
        """
        An epsilon-greedy base agent class.  The agent saves action-state value
        pairs as a nested dictionary of the form:
        {{'state0':{'action0':value,'action1':value...}},
         {'state1':{'action0':value,'action1':value...}}...}

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
        self.gw = gridworld
        self.alpha = alpha
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
        return {self.gw.current_loc : \
                    self.enumeratePossibleActions(self.gw.current_loc)}

    def reInitAgent(self):
        """
        Reinitializes the agent to have a clear history.
        """
        self.gw.initGridWorld()
        self.history = []

    def clearAgent(self):
        """
        Clears the agent's memory and history, and resets the gridWorld.
        """
        self.agent = self._makeNewAgent()
        self.reInitAgent()

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

    def stateVisitCount(self,state):
        """
        Counts how many times a state was visited by counting how many times it
        was left.

        Parameters
        ----------
        state : tuple of two integers
            The position on the game board to be evaluated.
        """
        try:
            vals = [x[1] for x in self.agent[state].values()]
        except KeyError:
            raise KeyError('Invalid state supplied to expectedStateValue.')
        return sum(vals)

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

    def findGreedyPath(self):
        """
        Find a path from start to goal using absolutely greedy move selection.
        """
        path = [self.gw.start_loc]
        move = self.returnGreedyPolicy(path[-1])
        new_loc = self.gw.validMove(move,path[-1])
        while new_loc not in path:
            path.append(new_loc)
            if new_loc == self.gw.goal_loc:
                break
            move = self.returnGreedyPolicy(path[-1])
            new_loc = self.gw.validMove(move,path[-1])
        return path

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
        Perform an update of the agent. The type of update is controlled by the
        finalQ function.
        """
        if len(self.history) < 2:
            return # not enough history yet
        S = self.history[-2][0]
        A = self.history[-2][1]
        R = self.history[-2][2]
        self.agent[S][A][0] += self.alpha * (R + \
                                    self.discount * self.finalQ() - \
                                    self.agent[S][A][0])
        self.agent[S][A][1] += 1 # update the visit count

    def finalQ(self):
        """
        Stub function for the final state of a single step update.  It is
        implemented by the various agent types (child classes).
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

    def printActionState(self):
        """
        Outputs the contents of the agent's action-state values.  If the
        gridworld the agent was trained on was large, the output of this command
        can be very large.
        """
        for k,v in self.agent.items():
            print('State: ',k,' Actions: ',v)

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

class nStepEpsilonGreedyAgent(epsilonGreedyAgent):
    """

    """
    def __init__(self,gridworld,n=1,alpha=0.5,epsilon=0.1,discount=1.0,name=None):
        """
        An epsilon-greedy base agent class.  The agent saves action-state value
        pairs as a nested dictionary of the form:
        {{'state0':{'action0':value,'action1':value...}},
         {'state1':{'action0':value,'action1':value...}}...}

        Parameters
        ----------
        gridworld : a gridworld object
            A pointer to a gridworld object that the agent can interact with.
        n : int, optional
            The number of steps to look foward when computing state values.
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
        super().__init__(gridworld=gridworld,
                         alpha=alpha,
                         epsilon=epsilon,
                         discount=discount,
                         name=name)
        self.n = n
        self.update_step = 0 # the next step in the history to update

    def reInitAgent(self):
        """
        Reinitializes the agent to have a clear history.
        """
        super().reInitAgent()
        self.update_step = 0

    def takeAction(self):
        """
        Picks an action from a given state using the epsilon-greedy algorithm.

        This is the agent's policy.  It is an epsilon-greedy policy.

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
            was_updated = self.updateQ(True) # always a non-terminal update
            if was_updated:
                self.update_step += 1
        # if you reached the terminal state
        if not not_terminal:
            self.checkForState(self.gw.getCurrentLoc())
            self.history.append((self.gw.getCurrentLoc(),0,0))
            while self.update_step < len(self.history):
                was_updated = self.updateQ(False) # always a terminal update
                if was_updated:
                    self.update_step += 1
        return not_terminal

    def updateQ(self,not_terminal):
        """
        Perform an n-step update of the agent using the final action-state given
        by finalQ.

        Parameters
        ----------
        not_terminal : bool
            Flag that is True when the agent has not yet reached the terminal
            state.

        Returns
        -------
        True if S,A is updated, False if no state is updated.
        """
        if len(self.history) < self.n + 1 and not_terminal:
            return False # not enough history yet
        S = self.history[self.update_step][0]
        A = self.history[self.update_step][1]
        G_n = self.computeReward() # this is G_n = R
        if self.update_step + self.n < len(self.history):
            val_discount = self.discount ** self.n
            G_n += val_discount * self.finalQ() # add in discount * end-Q
        # update the action-state value
        self.agent[S][A][0] += self.alpha * (G_n - self.agent[S][A][0])
        self.agent[S][A][1] += 1 # update the visit count
        return True

    def computeReward(self):
        """
        Computes the rewards accumulated and discounted along an n-step path
        from self.update_step.  If the n steps is longer than the available
        history, the rewards are computed up to the end of history.

        Returns
        -------
        The computed reward.
        """
        if self.update_step + self.n < len(self.history):
            return sum([self.history[self.update_step+k][2] * self.discount ** k
                     for k in range(self.n)])
        else: # nearing the end of the history
            return sum([self.history[self.update_step+k][2] * self.discount ** k
                     for k in range(len(self.history) - self.update_step - 1)])

#################################################
# SARSA Agents
#################################################
class sarsaAgent(epsilonGreedyAgent):
    """
    An agent that learns using the SARSA scheme and epsilson-greedy
    policy.
    """

    @staticmethod
    def __str__():
        return "SARSA Agent"

    def finalQ(self):
        """
        Sarsa agent final action-state for a single step update.

        Returns
        -------
        Q(S',A') for a single step update.
        """
        Sp = self.history[-1][0]
        Ap = self.history[-1][1]
        return self.agent[Sp][Ap][0]

class nStepSarsaAgent(nStepEpsilonGreedyAgent):
    """
    An agent that learns using the n-step SARSA scheme and epsilson-greedy
    policy.
    """

    @staticmethod
    def __str__():
        return "n-SARSA Agent"

    def finalQ(self):
        """
        Sarsa agent final action-state for a single step update.

        Returns
        -------
        Q(S',A') for a single step update.
        """
        Sp = self.history[self.update_step + self.n][0]
        Ap = self.history[self.update_step + self.n][1]
        return self.agent[Sp][Ap][0]

#################################################
# Q-learning Agents
#################################################
class QAgent(epsilonGreedyAgent):
    """
    An agent that learns using the Q-learning scheme and epsilson-greedy
    policy.
    """

    @staticmethod
    def __str__():
        return "Q Agent"

    def finalQ(self):
        """
        Q-learning agent final action-state for a single step update.

        Returns
        -------
        max_a Q(S',a) for a single step update.
        """
        Sp = self.history[-1][0]
        max_val = self.agent[Sp][0][0] # find the maximum Q(S',A')
        for val in self.agent[Sp].values():
            if val[0] > max_val:
                max_val = val[0]
        return max_val

class nStepQAgent(nStepEpsilonGreedyAgent):
    """
    An agent that learns using the n-step Q-learning scheme and epsilson-greedy
    policy.
    """

    @staticmethod
    def __str__():
        return "n-Q Agent"

    def finalQ(self):
        """
        Q-learning agent final action-state for a single step update.

        Returns
        -------
        max_a Q(S',a) for a single step update.
        """
        Sp = self.history[self.update_step + self.n][0]
        max_val = self.agent[Sp][0][0] # find the maximum Q(S',A')
        for val in self.agent[Sp].values():
            if val[0] > max_val:
                max_val = val[0]
        return max_val

#################################################
# Expected SARSA Agents
#################################################
class expectedSarsaAgent(epsilonGreedyAgent):
    """
    An agent that learns using the Expected SARSA scheme and epsilson-greedy
    policy.
    """

    @staticmethod
    def __str__():
        return "E-SARSA Agent"

    def finalQ(self):
        """
        Expected SARSA agent final action-state for a single step update.

        Returns
        -------
        E[Q(S',a)] for a single step update.
        """
        Sp = self.history[-1][0]
        max_val = self.agent[Sp][0][0]
        for val in self.agent[Sp].values():
            if val[0] > max_val:
                max_val = val[0]
        exp_prob = self.epsilon / len(self.agent[Sp].keys())
        expect = (1 - self.epsilon) * max_val + \
                 exp_prob * sum([x[0] for x in self.agent[Sp].values()])
        return expect

class nStepExpectedSarsaAgent(nStepEpsilonGreedyAgent):
    """
    An agent that learns using the n-step expected SARSA scheme and
    epsilson-greedy policy.
    """

    @staticmethod
    def __str__():
        return "n-E-SARSA Agent"

    def finalQ(self):
        """
        Expected SARSA agent final action-state for a single step update.

        Returns
        -------
        E[Q(S',a)] for a single step update.
        """
        Sp = self.history[self.update_step + self.n][0]
        max_val = self.agent[Sp][0][0]
        for val in self.agent[Sp].values():
            if val[0] > max_val:
                max_val = val[0]
        exp_prob = self.epsilon / len(self.agent[Sp].keys())
        expect = (1 - self.epsilon) * max_val + \
                 exp_prob * sum([x[0] for x in self.agent[Sp].values()])
        return expect

# some immediate testing
if __name__ == "__main__":
    from gridworld import gridWorld
    t = gridWorld()
    a = sarsaAgent(t)
