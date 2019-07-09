#!/usr/bin/env python3

import random

class KArmedBandit(object):
    """
    Implementation of the epsilon-greedy k-armed bandit testbed from
    Sutton and Barto's Reinforcement Learning: An Introduction (2nd Ed), Ch 2.
    """

    def __init__(self,k=10,steps=1000,
                      epsilon=0.0,initial_value=0.0,
                      track_rewards=True):
        """
        Parameters
        ----------
        k : int, optional
            The number of "bandits" to use. Default is 10.

        steps: int, optional
            The number of steps to take in a single trial.  Default is 2000.

        epsilon : float, optional
            The epsilon parameter.  0 <= epsilon <= 1.
            This parameter controls how often the epsilon-greedy algorithm
            explores sub-optimal actions instead of exploiting optimal actions.

        initial_value : float, optional
            The value to initialize the learner's value-list with.

        track_rewards : boolean, optional
            If true, this object will save all of the rewards it has seen over
            the course of its run.
        """
        # initialize some basic parameters
        self.k = abs(k)
        self.steps = steps
        self.epsilon = epsilon
        self.initial_value = initial_value
        self.track_rewards = track_rewards
        if self.track_rewards:
            self.rewards_seen = []
            self.optimal_action = []

        # initialize the learner and the bandits
        self.initialize_learner()
        self.initialize_bandits()

    def initialize_learner(self,epsilon=None,initial_value=None):
        """
        Initialize the learner.  The learner is two lists: values and number
        of times each action was taken.  The values are the learner's estimates
        of the bandit's reward pay out.

        Parameters
        ----------
        epsilon : float, optional
            The epsilon parameter.  0 <= epsilon <= 1.
            This parameter controls how often the epsilon-greedy algorithm
            explores sub-optimal actions instead of exploiting optimal actions.

        initial_value : float, optional
            The value to initialize the learner's value-list with.
        """
        if epsilon: # user supplies an epsilon value
            self.epsilon = epsilon
        if initial_value: # user supplies an initial_value
            self.initial_value = initial_value
        self.values = [self.initial_value for _ in range(self.k)]
        self.counts = [0 for _ in range(self.k)]
        if self.track_rewards: # reset the rewards_seen attribute
            self.rewards_seen = []
            self.optimal_action = []

    def initialize_bandits(self,mean=0,stdev=1):
        """
        Initialize the bandits with means and standard deviations drawn from
        a normal distribution.

        Parameters
        ----------
        mean : float, optional
            Mean value of the normal distribution to draw the bandit parameters
            from.

        stdev : float, optional
            Standard deviation of the normal distribution to draw the bandit
            parameters from.
        """
        mu = random.gauss(mean,stdev)
        self.bandits = [Bandit(mu,1)]
        # Optimal bandit is the best bandit to use, the training algorithm
        # should not be aware of this parameter.  It is used to compute the
        # % optimal actions
        self.optimal_bandit = 0
        for idk in range(1,self.k):
            mu = random.gauss(mean,stdev)
            self.bandits += [Bandit(mu,1)]
            if mu > self.bandits[self.optimal_bandit].mean:
                self.optimal_bandit = idk

    def reward(self,action):
        """
        Return a reward for a given action.

        Parameters
        ----------
        action : int
            Which bandit to use.

        Returns
        -------
        reward : float
            The reward given by the stored reward model.
        """
        return self.bandits[action].reward()

    def update_counts(self,action):
        """
        Update the count of which actions has been taken.

        Parameters
        ----------
        action : int
            Which bandit to use.
        """
        try:
            self.counts
        except AttributeError:
            print('The learner has not been initialized.')
        else:
            self.counts[action] += 1

    def update_values(self,action,reward):
        """
        Update the value of which action has been taken.  Also calls the
        update_counts method.

        Parameters
        ----------
        action : int
            Which bandit to use.

        reward : float
            The reward which the action returned.
        """
        self.update_counts(action)
        # self.update_counts checks that the learner exists, don't do it here
        self.values[action] += \
                (reward - self.values[action]) / self.counts[action]

    def step(self):
        """
        Take a time step in the reinforcement learning process.
        """
        if random.uniform(0,1) < self.epsilon: # explore
            action = random.choice(range(self.k))
        else: # exploit
            action = amax(self.values)
        if action == self.optimal_bandit:
            optimal = 1
        else:
            optimal = 0
        reward = self.reward(action)
        self.update_values(action,reward)
        if self.track_rewards:
            self.rewards_seen += [reward]
            self.optimal_action += [optimal]

    def train_learner(self,steps=None):
        """
        Runs a simulation of fix length.

        Parameters
        ----------
        steps : int, optional
            The number of steps to take in this simulation.  If no value is
            given by the user this method uses the value supplied when the
            class instance was created.
        """
        if not steps:
            steps = self.steps
        for _ in range(steps):
            self.step()


class Bandit(object):
    """
    A single one-armed bandit (i.e. a slot machine).

    The rewards from this bandit are normally distributed with mean and
    standard deviation as supplied by the user.
    """

    def __init__(self,mean=0,stdev=1):
        """
        Parameters
        ----------
        mean : float, optional
            Mean of the normal distribution sampled by this bandit.

        stdev : float, optional
            Standard deviation of the normal distribution sampled by the bandit.


        """
        self.mean = mean
        self.stdev = stdev

    def reward(self):
        """
        Receive a reward from the bandit.

        Parameters
        ----------
        None.

        Returns
        -------
        A reward sampled from the bandit's normal distribution.

        """
        return random.gauss(self.mean,self.stdev)

def amax(list_of_values):
    the_max = list_of_values[0]
    arg_max = 0
    for idx,val in enumerate(list_of_values[1:]):
        if val > the_max:
            the_max = val
            arg_max = idx + 1
    return arg_max
