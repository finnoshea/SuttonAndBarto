#! /usr/bin/env python3
"""
A solution to Example 4.2 in Sutton and Barto.
"""

import math
import time
from numpy.random import poisson
import matplotlib.pyplot as plt

class JacksCarRental(object):
    """

    """

    def __init__(self,num_cars=None):
        """

        """
        # hardcode some problem features
        self.gamma = 0.9 # discount rate
        self.max_move = 5 # the maximum number of cars that can be moved
        if not num_cars:
            self.num_cars = 20 # default problem has 20 cars per lot max
        else:
            self.num_cars = num_cars
        # initialize the problem
        self.states = self.init_states()
        self.policies = self.init_policies()

    def key_convert(self,x):
        """
        Takes either a string or a list and converts to the other.
        [1,3] -> '1,3'
        '1,3' -> [1,3]

        Parameters
        ----------
        x : either a string or list
            The item to be converted

        Returns
        -------
        The converted item
        """
        if isinstance(x,str):
            return [a for a in map(int,x.split(','))]
        elif isinstance(x,list):
            return ','.join([a for a in map(str,x)])
        else:
            print('Unexpected type encountered, input must be str or list(int)')
            return None

    def init_states(self,value=0):
        """
        Initializes a dictionary of the states for the problem with the
        given values.  This is the value function.

        Parameters
        ----------
        value : float or int
            The initialization state for the problem.  Usually arbitrary.

        Returns
        -------
        A dictionary of the states, all with value 'value'.
        """
        adict = {}
        for i in range(self.num_cars+1):
            for j in range(self.num_cars+1):
                adict[self.key_convert([i,j])] = value
        return adict

    def init_policies(self,value=0):
        """
        Initializes a dictionary of the policies for the problem with the
        given values.  It is a dictionary of scalar values.

        Parameters
        ----------
        value : float or int
            The initialization state for the problem.

        Returns
        -------
        A dictionary of the states, all with value 'value'.
        """
        adict = {}
        for i in range(self.num_cars+1): # cars at first location
            for j in range(self.num_cars+1): # cars at the second location
                the_key = self.key_convert([i,j])
                adict[the_key] = value
        return adict

    def poisson(self,average=5,n=5):
        """
        Compute the probability drawn from a poisson distribution.

        Parameters
        ----------
        average : float
            The mean number of whatever event over a large interval.
        n : int
            The number of events for which to compute the probability.

        Returns
        -------
        A probability between 0 and 1 (inclusive).

        """
        return (math.exp(-1*average) * average ** n) / math.factorial(n)

    def joint_poisson(self,m,n,u,v):
        """
        Computes the joint probability of 4 independent Poisson-distributed
        events: that there will be m rental requests at lot 0, that there will
        be n rental requests at lot 1, that there will be u returns at lot 0,
        and that there will be v returns at lot 1.

        Parameters
        ----------
        m : int
            The number of rental requests at lot 0.
        n : int
            The number of rental requests at lot 1.
        u : int
            The number of returns to lot 0
        v : int
            The number of returns to lot 1

        Returns
        -------
        float
            The probability of the events occuring jointly.

        """
        avg_rentals_one = 3
        avg_rentals_two = 4
        avg_returns_one = 3
        avg_returns_two = 2
        return self.poisson(avg_rentals_one,m) * \
               self.poisson(avg_rentals_two,n) * \
               self.poisson(avg_returns_one,u) * \
               self.poisson(avg_returns_two,v)

    def norm_poi(self,average=5,max_n=5,n=5):
        """
        Compute the probability drawn from a poisson distribution that is
        normalized such that the sum of probabilities up to max_n is 1. This is
        required for the Markov Decision Process formalism.

        The normalization has a closed form that uses incomplete and complete
        gamma functions.  Just do the sum.

        Parameters
        ----------
        average : float
            The mean number of whatever event over a large interval.
        max_n : int
            The maximum allowed value for the sampling of the distribution.
        n : int
            The number of events for which to compute the probability.

        Returns
        -------
        A probability between 0 and 1 (inclusive).
        """
        norm = sum([(average ** nn / math.factorial(nn)) \
                    for nn in range(max_n+1)])
        return (average ** n) / (norm * math.factorial(n))

    def iterative_policy_eval(self,tolerance=1e-3,verbose=False):
        """
        Performs iterative policy evaluation following the Sutton and Barto
        algorithm on page 75 in Chapter 4.

        Parameters
        ----------
        tolerance : float
            Terminates the iteration when the largest difference between a state
            and its updated value is less than tolerance.
        verbose : boolean
            A flag telling the method to print out some information during run
            time.

        Returns
        -------
        The updated state values in a dictionary.
        """
        delta = 2 * tolerance
        V = self.states.copy() # create a local dictionary of the states
        iter = 0
        while delta > tolerance:
            if verbose and not iter % 10:
                print('Iterative Policy Evaluation Iteration: {:d}'\
                      .format(iter))
            delta = 0.0 # default to leaving the while loop
            for s,v in V.items():
                # s is the initial state, v is the initial value
                new_V = 0
                action = self.policies[s]
                sps = self.enumerate_final_states(s,action)
                for sp,pr in sps.items():
                    new_V += pr[0]*(pr[1] + self.gamma * V[sp])
                V[s] = new_V
                delta = max(delta,abs(v - V[s]))
            iter += 1
        if verbose:
            print('Policy Evaluation has converged.')
        return V

    def enumerate_final_states(self,s,a):
        """
        Computes the final states for a given initial state and action. The
        order of events is:
        0) Move cars (take action)
        1) Rent cars out
        2) Receive returned cars

        Parameters
        ----------
        s : str
            The initial state.
        a : int
            The action to be taken: the number of cars moved from lot 0 to lot 1
            negative numbers are valid.

        Returns
        -------
        A dictionary of possible final states (the key) with a list of
        probability and reward as the value.
        """
        base_reward = 10
        move_cost = 2
        s = self.key_convert(self.take_action(s,a)) # move cars
        final_states = {}
        norm = 0 # normalization constant
        for m in range(self.num_cars+1): # rental requests at lot 0
            pm = self.poisson(3,m) # probability of m rental requests
            if m > s[0]: # more rental requests than cars
                m = s[0] # rent out all the cars
            for n in range(self.num_cars+1): # rental requests at lot 1
                pn = self.poisson(4,n)
                if n > s[1]: # more rental requests than cars
                    n = s[1] # rent out all the cars
                for u in range(self.num_cars+1): # cars returned at lot 0
                    pu = self.poisson(3,u)
                    for v in range(self.num_cars+1): # cars returned at lot 1
                        pv = self.poisson(2,v)
                        p = pm * pn * pu * pv
                        sp = [s[0] - m + u, s[1] - n + v]
                        if sp[0] > self.num_cars:
                            sp[0] = self.num_cars
                        if sp[1] > self.num_cars:
                            sp[1] = self.num_cars
                        sp = self.key_convert(sp)
                        norm += p
                        reward = self.compute_reward(m,n,a)
                        try:
                            final_states[sp][0] += p
                            final_states[sp][1] += p * reward
                        except KeyError:
                            final_states[sp] = [0,0]
                            final_states[sp][0] = p
                            final_states[sp][1] = p * reward
        # perform the required normalization for mean(reward) and make sure
        # the probability of going to any sp with any r sums to 1:
        for key,value in final_states.items():
            # normalize the rewards (this is sum over weights in a mean)
            # this is the sum over r
            final_states[key][1] = final_states[key][1] / final_states[key][0]
            # normalize the total probability
            final_states[key][0] = final_states[key][0] / norm

        return final_states


    def sum_over_r(self,state,state_value,action):
        """
        Performs the sum over reward (r) in the formula for iterative policy
        evaluation.

        Parameters
        ----------
        state : str
            The state, s', for which the sum over r is being performed.
        statve_value: float
            The value of the state s'.
        action: int
            The action taken to arrive at state s'.

        Returns
        -------
        A value to add to V(s) as given in the formula.

        """
        temp_V = 0
        max_m,max_n = map(int,state.split(',')) # max cars to rent
        # enumerate the ways cars can be rented
        for way in self.enumerate_car_rentals(state): # perform the sum over r
            m,n = map(int,way.split(','))
            temp_V += self.norm_poi(3,max_m,m) * \
                     self.norm_poi(4,max_n,n) * \
                     (self.compute_reward(m,n,action) +
                      self.gamma * state_value)
        return temp_V

    def policy_update(self,is_stable,verbose=False):
        """
        Performs a policy update step within the policy improvement loop.

        Parameters
        ----------
        is_stable : bool
            Boolean saying whether or not the current policy is stable (true)
            or not (false).
        verbose : boolean
            A flag telling the method to print out some information during run
            time.

        Returns
        -------
        A tuple containing:
            is_stable, which might have changed.
            The updated policy dictionary.

        """
        old_policies = self.policies.copy()
        new_policies = {}
        for idx,s in enumerate(self.states.keys()):
            if verbose and not idx % 10:
                print('States evaluated in policy update: {:d} / {:d}'\
                      .format(idx,len(self.states)))
            actions = {}
            for a in self.enumerate_actions(s):
                sps = self.enumerate_final_states(s,a)
                actions[a] = sum([pr[0]*(pr[1] + self.gamma * self.states[sp])
                                  for sp,pr in sps.items()])
            max_a, _ = self.argmax(actions)
            if max_a != old_policies[s]:
                is_stable = False
                new_policies[s] = max_a
            else:
                new_policies[s] = old_policies[s]
        if verbose:
            print('Policy update is stable: {}'.format(is_stable))
        return is_stable, new_policies

    def policy_improvement_step(self,is_stable,verbose=False):
        """
        Performs a single step of policy improvement.  Step 3 in the formula of
        Sutton and Barto on page 80.

        Parameters
        ----------
        is_stable : bool
            Boolean saying whether or not the current policy is stable (true)
            or not (false).
        verbose : boolean
            A flag telling the method to print out some information during run
            time.

        Returns
        -------
        is_stable, which might have changed. States and policies are updated in
        place.
        """
        is_stable, self.policies = self.policy_update(is_stable=is_stable,
                                                      verbose=verbose)
        if not is_stable: # if the policy isn't yet stable
            self.states = self.iterative_policy_eval(verbose=verbose) # update states
        return is_stable

    def policy_improvement(self,verbose=False):
        """
        Perform the policy imporovement loop.

        Parameters
        ----------
        verbose : boolean
            A flag telling the method to print out some information during run
            time.

        Returns
        -------
        Nothing. States and policies are updated in place.
        """
        is_stable = False
        t0 = time.time() # start time
        iteration = 0
        while not is_stable:
            is_stable = True
            iteration += 1
            t1 = time.time()
            is_stable = self.policy_improvement_step(is_stable=is_stable,
                                                     verbose=verbose)
            t2 = time.time()
            if verbose:
                print('Policy improvement step {:d} took {:4.3f} seconds'\
                      .format(iteration,t2-t1))

    def enumerate_actions(self,state):
        """
        Enumerates the possible actions allowed from a given state.  This
        function does not allow the number of cars in a lot to exceed num_cars.
        The maximum number of cars that can be moved is also limited by the
        max_move attribute.

        Example: state = '1,2'
        Returns: [-2,-1,0,1]

        Parameters
        ----------
        state : str
            The state for which the possible actions should be enumerated.

        Returns
        -------
        A list of allowed actions.
        """
        state = self.key_convert(state)
        mini = -1*(self.num_cars - state[0]) if (self.num_cars - state[0]) \
            <= state[1] else -1*state[1]
        mini = mini if abs(mini) <= self.max_move else -1*self.max_move
        maxi = (self.num_cars - state[1]) if (self.num_cars - state[1]) \
            <= state[0] else state[0]
        maxi = maxi if abs(maxi) <= self.max_move else self.max_move
        return [x for x in range(mini,maxi+1)]

    def take_action(self,initial_state,action):
        """
        Converts an initial state in to a final state using action.  Positive
        action moves cars from state 0 to to state 1.  Reducing the number of
        cars in either state 0 or state 1 to below zero or above 20 is not
        allowed, but this is controled for in the actions parameter.  Ad-hoc
        use of the take_action method may result in unintended consequences.

        Example: initial_state = '1,3'
                 action = -1
        Returns: final_state = '2,2'

        Parameters
        ----------
        initial_state : str
            The initial state in string form.
        action : int
            The number of cars to move.

        Returns
        -------
        A final state in string form.

        """
        state = self.key_convert(initial_state)
        state[0] -= action
        state[1] += action
        return self.key_convert(state)

    def compute_reward(self,m,n,a):
        """
        Compute the reward for renting 'm' cars from lot 0 (state 0), 'n cars
        from lot 1 (state 1) and moving 'a' cars between lots.

        Parameters
        ----------
        m : int
            The number of cars rented from lot 0
        n : int
            The number of cars rented from lot 1
        a : int
            The number of cars moved between the two lots.

        Returns
        -------
        An integer reward value.
        """
        base_reward = 10
        move_cost = 2
        return base_reward * (m + n) - move_cost * abs(a)

    def enumerate_car_rentals(self,state):
        """
        Enumerates the number of ways that cars can be rented given the state.
        If the state is 'm,n' there are (m+1)*(n+1) ways cars can be rented.

        Example: state = '2,1'
        Returns: ['0,0', '0,1', '1,0', '1,1', '2,0', '2,1']

        Parameters
        ----------
        state : str
            The state of the two rental lots.

        Returns
        -------
        A list of strings of the ways cars can be rented.
        """
        ways = []
        state = self.key_convert(state)
        for im in range(state[0]+1):
            for jn in range(state[1]+1):
                ways += [str(im)+','+str(jn)]
        return ways

    @staticmethod
    def argmax(dictionary):
        """
        Performs argmax on a dictionary and returns the key associated with the
        maximum value.

        Parameters
        ----------
        dictionary : a dictionary
            The dictionary to be argmax'd, the values must be numbers that can
            be compared.  The keys can be any valid key.

        Returns
        -------
        A tuple consisting of (key,value) that are argmax.
        """
        max_key = next(iter(dictionary.keys())) # arbitrary initialization
        max_val = dictionary[max_key] # arbitrary initialization
        for key,value in dictionary.items():
            if value > max_val:
                max_key = key
                max_val = value
        return max_key, max_val

    def plot_policy(self,title=''):
        """
        Plots the policy as an image using matplotlib.
        """
        # initialize the array
        array = [[0 for _ in range(self.num_cars + 1)] for _ in
                                        range(self.num_cars + 1)]
        for key,value in self.policies.items():
            loc = self.key_convert(key)
            array[loc[0]][loc[1]] = value
        fig,ax = plt.subplots(1)
        pict = ax.imshow(array,origin='lower')
        ax.set_xlabel('# of cars in first location')
        ax.set_ylabel('# of cars in second location')
        ax.set_title(title)
        ax.set_xticks([x for x in range(self.num_cars + 1)])
        ax.set_yticks([x for x in range(self.num_cars + 1)])
        ax.set_yticklabels([str(x) for x in range(self.num_cars + 1)])
        ax.set_xticklabels([str(x) for x in range(self.num_cars + 1)])
        fig.colorbar(pict, ax=ax)
        plt.show(block=False)

    def plot_states(self,title=''):
        """
        Plots the states as an image using matplotlib
        """
        # initialize the array
        array = [[0 for _ in range(self.num_cars + 1)] for _ in
                                        range(self.num_cars + 1)]
        for key,value in self.states.items():
            loc = self.key_convert(key)
            array[loc[0]][loc[1]] = value
            fig,ax = plt.subplots(1)
            pict = ax.imshow(array,origin='lower')
            ax.set_xlabel('# of cars in first location')
            ax.set_ylabel('# of cars in second location')
            ax.set_title(title)
            ax.set_xticks([x for x in range(self.num_cars + 1)])
            ax.set_yticks([x for x in range(self.num_cars + 1)])
            ax.set_yticklabels([str(x) for x in range(self.num_cars + 1)])
            ax.set_xticklabels([str(x) for x in range(self.num_cars + 1)])
            fig.colorbar(pict, ax=ax)
            plt.show(block=False)
