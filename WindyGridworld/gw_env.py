#! /usr/bin/env python3
"""
An environment for running episodes and collecting data for the various grid
world problems of chapter 6 of Sutton and Barto.
"""

from gridworld import gridWorld, kingWorld
from gw_agents import sarsaAgent, QAgent, expectedsarsaAgent
import matplotlib.pyplot as plt


class GWEnv(object):
    """

    """
    def __init__(self,gridworld=gridWorld,size=(7,10),start_loc=(3,0),
                 goal_loc=(3,7),max_moves=None,agent=sarsaAgent,alpha=0.5,
                 epsilon=0.1,discount=1.0,name=None):
        """
        An environment where the agent learns on gridworld.

        Parameters
        ----------
        gridworld : a gridworld object
            A pointer to a gridworld object that the agent can interact with.
        size : list or tuple of two integers
            Specifies the size of the grid world in rows and columns.
        start_loc : list or tuple of two integers
            Specifies the location of the starting grid point.
        goal_loc : list or tuple of two integers
            Specifies the location of the goal grid point.
        max_moves : int
            The maximum number of allowed moves before the episode is forced to
            end.  Default is None.
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
        # create an environment
        gw = gridworld.__call__(size,start_loc,goal_loc,max_moves)
        self.agent = agent.__call__(gw,alpha,epsilon,discount,name)

        self.summary_stats = []

    def runEpisode(self):
        """
        Runs an episode of the grid world search by the agent and adds a
        summary of the result to self.summary_stats.
        """
        v = True
        self.agent.reInitAgent()
        while v: # let the agent move until it satisfies an exit condition
            v = self.agent.takeAction()
        self.summary_stats.append(self.agent.summarizeHistory())

    def runNEpisodes(self,n=100):
        """
        Runs n episodes of the current grid world and agent.

        Parameters
        ----------
        n : int
            The number of episodes to run.
        """
        iter = 0
        while iter < n:
            self.runEpisode()

    def runNTimeSteps(self,n=8000):
        """
        Runs as many epsiodes as required to produce n total time steps across
        all the episodes.  As the number of steps to finish any episode is not
        fixed, this function will stop after the epsiode that increases the
        total number of time steps to greater than or equal to n.

        Parameters
        ----------
        n : int
            The number of time steps to experience.
        """
        ts = 0
        while ts < n:
            self.runEpisode()
            ts += self.summary_stats[-1][0]

    def plotEpisodesVsTimeSteps(self):
        if not self.summary_stats:
            print('You have not run any episodes yet.')
            return
        # format the data for plotting
        accum_time_steps = [self.summary_stats[0][0]]
        rewards = [self.summary_stats[0][1]]
        episodes = [1]
        for ep,episode in enumerate(self.summary_stats[1:]):
            episodes.append(ep+2)
            rewards.append(episode[1])
            accum_time_steps.append(accum_time_steps[-1] + episode[0])
        # compute the average number of steps per episode for the last
        # 20% of episodes
        num_last_eps = int(0.2 * len(self.summary_stats))
        average = sum([x[0] for x in self.summary_stats[-num_last_eps:]]) \
                    / num_last_eps

        plt.plot(accum_time_steps,episodes)
        plt.ylabel('Episodes')
        plt.xlabel('Time steps')
        plt.title('Avg number of steps at the end: {:3.2f}'.format(average))
        plt.show()

    def plotStateValues(self):
        """
        Creates a matplotlib image plot of the state value for each state in
        the agent's experience.
        """
        gw = []
        for i in range(self.agent.gw.size[0]):
            row = []
            for j in range(self.agent.gw.size[1]):
                row.append(self.agent.expectedStateValue((i,j)))
                l = (i,j)
                p = self.agent.gw.allowed_actions[
                                self.agent.returnGreedyPolicy(l)].__call__()
                plt.arrow(l[1],l[0],p[1],p[0],
                          length_includes_head=True,head_width=0.1)
            gw.append(row)
        plt.imshow(gw,origin='lower')
        plt.show()

if __name__ == "__main__":
    t = GWEnv(gridworld=gridWorld,size=(2,2),start_loc=(0,0),goal_loc=(1,1),
              agent=QAgent)
    t.runNTimeSteps(n=100000)
    # t.plotEpisodesVsTimeSteps()
    t.plotStateValues()
