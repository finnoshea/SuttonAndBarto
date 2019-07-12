#! /usr/bin/env python3
"""
Example 4.3 in Sutton and Barto
"""

from random import choice as rc
import matplotlib.pyplot as plt

class GamblersProblem(object):
    """

    """

    def __init__(self,ph=0.4,max_capital=100):
        """
        Class for solving the Gambler's Problem in Sutton and Barto Chapter 4.
        Uses value iteration to find the optimal policy.

        If the gambler reaches max_capital, he wins.  If the gambler reaches
        0 capital, he loses.

        Parameters
        ----------
        ph : float
            Value, 0 < ph < 1, for the probability of seeing a heads on a coin
            flip.
        max_capital : int
            Value of capital that signifies that the gambler has won.  The
            allowed states are from 1 to max_captial - 1.

        """
        self.ph = ph
        self.max_capital = max_capital
        self.gamma = 1.0 # discount factor for future rewards
        self.vi_tol = 1e-11 # tolerance for value iteration completion

        self.states = self.init_states()
        self.policies = self.init_policies()

    def init_states(self):
        """
        Initializes the states with zeros.
        """
        return [0 for _ in range(self.max_capital)]

    def init_policies(self):
        """
        Initializes the policies with zeros.
        """
        return [0 for _ in range(self.max_capital)]

    def p_and_r(self,s,a):
        """
        Computes p(s',r|s,a) and the reward for all possible s' and r given s
        and a.

        Parameters
        ----------
        s : int
            State of the gambler's capital.
        a : int
            Gambler's action, i.e. how much he wagers that a heads will come up.

        Returns
        -------
        A dictionary with two keys, each with a value [p,r]:
            s-a : [1-self.ph,0] # tails is seen, p=1-ph, no reward
            s+a :   [self.ph,r] # heads is seen, p = ph, possible reward.
        """
        if s+a == self.max_capital:
            r = 1 # only rewarded for winning
        else:
            r = 0
        # the gambler can only win on a heads
        return {s-a: [1-self.ph,0], s+a: [self.ph,r]}

    def enumerate_actions(self,s):
        """
        Enumerates all the actions (bets) allowed from state s.  Lazy.

        Parameters
        ----------
        s : int
            State of the gambler's capital.

        Returns
        -------
        A list of the allowed actions.
        """
        for x in range(min(s+1,self.max_capital - s + 1)):
            yield x

    def compute_action_value(self,sa_dict):
        """
        Computes the action-value for a given dictionary of the form s' : [p,r].
        I.e. it performs the double sum in
        Sum_s' Sum_r p(s',r|s,a) * [r + gamma * V(s')]

        For this problem there is only one r, so that sum drops out.

        Parameters
        ----------
        sa_dict : dictionary
            A dictionary of the form s' : [p,r].

        Returns
        -------
        The scalar value of the double sum.
        """
        total = 0
        for sp,p in sa_dict.items():
            # gambler loses or wins - no future states in either case
            if sp == 0 or sp == self.max_capital:
                total += p[0] * p[1]
            else: # normal situation
                total += p[0] * (p[1] + self.gamma * self.states[sp])
        return total

    def keep_best_actions(self,q_max,a_max,q,a):
        """
        Determines if a new action-value is greater than q_max and, if so,
        replaces q_max with q and the list of actions a_max with [a].  If q is
        equal to q_max, it adds a to the list a_max.

        Parameters
        ----------
        q_max : float
            The value of the highest reward action-value.
        a_max : list of integers
            A list of the actions that produce q_max reward.
        q : float
            The new action-value to compare to q_max.
        a : int
            The new action associated with q.

        Returns
        -------
        q_max and a_max, possibly updated.
        """
        if q - q_max > self.vi_tol:
            q_max = q
            a_max = [a]
        elif abs(q - q_max) < self.vi_tol:
            a_max += [a]
        return q_max, a_max

    def value_iteration(self):
        """
        Performs value iteration following the formula on page 83 of Sutton and
        Barto.

        Parameters
        ----------
        None.

        Returns
        -------
        The value of delta, which will be less than self.vi_tol.
        """
        delta = 2 * self.vi_tol
        while delta >= self.vi_tol:
            delta = 0 # default is stop looping
            for s,v in zip(range(1,self.max_capital),self.states[1:]):
                q_max = 0
                a_max = []
                for a in self.enumerate_actions(s):
                    # compute the action-value
                    q = self.compute_action_value(self.p_and_r(s,a))
                    q_max, a_max = self.keep_best_actions(q_max,a_max,q,a)
                self.states[s] = q_max
                # self.policies[s] = rc(a_max) # randomly pick a best action
                self.policies[s] = min(a_max) # pick the smallest action
                delta = max(delta,abs(v - self.states[s]))
        return delta

    def plot_policies(self,title=''):
        plt.bar(range(len(t.policies)),t.policies)
        plt.xlabel('Capital')
        plt.ylabel('Final Policy (stake)')
        plt.title(title)
        plt.show(block=False)

    def plot_states(self,title=''):
        plt.plot(self.states)
        plt.xlabel('Capital')
        plt.ylabel('Value Estimates')
        plt.title(title)
        plt.show(block=False)
