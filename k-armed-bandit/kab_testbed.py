#! /usr/bin/env python3

import matplotlib.pyplot as plt
from karmedbandit import KArmedBandit

class KABTestBed(KArmedBandit):
    """
    Testbed for running multiple independent k-armed-bandit simulations.
    Follows Sutton and Barto, Chapter 2.
    """

    def __init__(self,runs=2000,k=10,steps=1000,
                     epsilon=0.0,initial_value=0.0,
                     track_rewards=True):
        """
        Parameters
        ----------
        runs : int, optional
            Number of independent runs to perform.

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
        self.runs = runs
        super().__init__(k,steps,epsilon,initial_value,track_rewards)

    def run_n_simulations(self,runs=None):
        """
        Runs a fixed number of independent simulations.

        Parameters
        ----------
        runs : int, optional
            The number of simulations to run.  The default value is set when the
            class instance is created.
        """
        if not runs:
            runs = self.runs
        if self.track_rewards:
            self.run_rewards = []
            self.run_optimals = []
        for _ in range(runs):
            # initialize the simulation
            self.initialize_learner()
            self.initialize_bandits()
            self.train_learner()
            if self.track_rewards:
                self.run_rewards += [self.rewards_seen]
                self.run_optimals += [self.optimal_action]

    def show_average_reward(self):
        """
        Uses matplotlib to plot the average reward as a function of step number.
        Follow Sutton and Barto figure 2.2.
        """
        if not self.track_rewards:
            print('You must run the simulations with track_rewards enabled.')
            return None
        ax1 = plt.subplot(2,1,1)
        ax1.plot(pop_mean(transpose(self.run_rewards)),'-b')
        ax1.set_ylabel('Average reward')
        ax1.set_title(r'$\epsilon$ = {:3.2f}'.format(self.epsilon) + \
            ' and initial_value = {:3.2f}'.format(self.initial_value))
        ax1.axis([-10,self.steps+1,0,1.55])
        ax2 = plt.subplot(2,1,2)
        ax2.plot(pop_mean(transpose(self.run_optimals)),'-b')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Fraction Optimal Action')
        ax2.axis([-10,self.steps+1,0,1])
        plt.show()

def transpose(list_of_lists):
    """
    Transposes a list of lists.
    """
    return [y for y in map(list,zip(*list_of_lists))]

def pop_mean(list_of_lists):
    """
    Returns the mean along the first axis of a list of lists.
    """
    return [1.0*sum(x)/len(x) for x in list_of_lists]
