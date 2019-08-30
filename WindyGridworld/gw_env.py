#! /usr/bin/env python3
"""
An environment for running episodes and collecting data for the various grid
world problems of chapter 6 of Sutton and Barto.
"""

from gridworld import gridWorld, kingWorld, windyGridWorld, windyKingWorld
#  from gw_agents import sarsaAgent, QAgent, expectedSarsaAgent
from gw_agents import nStepSarsaAgent, nStepQAgent, nStepExpectedSarsaAgent
import matplotlib.pyplot as plt


class GWEnv(object):
    """

    """
    def __init__(self,gridworld=gridWorld,size=(7,10),start_loc=(3,0),
                 goal_loc=(3,7),random_start=True,max_moves=None,
                 agent=nStepQAgent,n=1,alpha=0.5,epsilon=0.1,
                 discount=1.0,name=None):
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
        random_start : bool
            Whether or not to randomize the start location of each episode.
        max_moves : int
            The maximum number of allowed moves before the episode is forced to
            end.  Default is None.
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
        # create an environment
        gw = gridworld.__call__(size,start_loc,goal_loc,max_moves)
        self.agent = agent.__call__(gridworld=gw,
                                    n=n,
                                    alpha=alpha,
                                    epsilon=epsilon,
                                    discount=discount,
                                    name=name)

        # summary_stats is a list of tuples (steps,total reward) for all the
        # episodes seen during this instance of the environment
        self.summary_stats = []

    def reInitEnv(self):
        """
        Reinitializes the environment and the agent.
        """
        self.agent.clearAgent()
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
            iter += 1

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
        accum_time_steps = [0,self.summary_stats[0][0]]
        rewards = [self.summary_stats[0][1]]
        episodes = [0,1]
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
        plt.axis([0,accum_time_steps[-1],0,episodes[-1]])
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
        visits = []
        for i in range(self.agent.gw.size[0]):
            row = []
            vrow = []
            for j in range(self.agent.gw.size[1]):
                l = (i,j)
                try:
                    sv = self.agent.expectedStateValue(l)
                    p = self.agent.gw.allowed_actions[
                                    self.agent.returnGreedyPolicy(l)].__call__()
                    vts = self.agent.stateVisitCount(l)
                    plt.arrow(l[1],l[0],p[1],p[0],
                              length_includes_head=True,head_width=0.1)
                except KeyError:
                    sv = 0
                    vts = 0
                row.append(sv)
                vrow.append(vts)
            gw.append(row)
            visits.append(vrow)
        plt.imshow(gw,origin='lower')
        for i,row in enumerate(visits):
            for j,elem in enumerate(row):
                plt.text(j,i-0.5,str(elem),fontsize=10,color='red')
        plt.show()

    def plotGreedyPath(self):
        fig = plt.figure()
        ax = fig.gca()
        ax.set_xticks([x - 0.5 for x in range(self.agent.gw.size[1])])
        ax.set_yticks([x - 0.5 for x in range(self.agent.gw.size[0])])
        plt.axis([-0.5,self.agent.gw.size[1] - 0.5,
                  -0.5,self.agent.gw.size[0] - 0.5])
        plt.plot([0,1,2,3,4,5,6,7,8,9,9,9,9,9,8,7],
                 [3,3,3,3,4,5,6,6,6,6,5,4,3,2,2,3],'--b')
        greedy_path = self.agent.findGreedyPath()
        plt.plot([x[1] for x in greedy_path],
                 [x[0] for x in greedy_path],'-r')
        plt.grid(linestyle='-',color='k')
        plt.text(self.agent.gw.start_loc[1]-0.25,
                 self.agent.gw.start_loc[0]-0.25,'S',fontsize=30)
        plt.text(self.agent.gw.goal_loc[1]-0.25,
                 self.agent.gw.goal_loc[0]-0.25,'G',fontsize=30)
        plt.tick_params(
            axis='both',       # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left=False,        # ticks along the left edge are off
            right=False,       # ticks along the right edge are off
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelleft=False,   # labels along the left edge are off
            labelbottom=False) # labels along the bottom edge are off
        try: # add wind values to the plot if they exist
            for x,c in enumerate(self.agent.gw.wind_cols):
                plt.text(x,-0.75,str(c))
            for y,c in enumerate(self.agent.gw.wind_rows):
                plt.text(-0.75,y,str(c))
        except AttributeError:
            pass
        plt.title('Greedy path steps: {:d}'.format(len(greedy_path)-1))
        plt.show()

if __name__ == "__main__":
    # t = GWEnv(gridworld=windyKingWorld,agent=QAgent,max_moves=None)
    # t.runNTimeSteps(n=8000)
    # t.plotEpisodesVsTimeSteps()
    # t.plotGreedyPath()
    # t.plotStateValues()

    t = GWEnv(gridworld=gridWorld,agent=nStepExpectedSarsaAgent,max_moves=None,
              n=1,random_start=False)
    t.runNTimeSteps(n=8000)
    t.plotEpisodesVsTimeSteps()
    t.plotGreedyPath()
