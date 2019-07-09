#! /usr/bin/env python3
"""
A solution to Example 4.2 in Sutton and Barto.
"""

import math

class JacksCarRental(object):
    """

    """

    def __init__(self,num_cars=None):
        """

        """
        # hardcode some problem features
        self.gamma = 0.9 # discount rate
        if not num_cars:
            self.num_cars = 20 # default problem has 20 cars per lot max
        else:
            self.num_cars = num_cars
        # initialize the problem
        self.states = self.init_states()
        self.actions = self.init_actions()

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

    def init_actions(self,value=1):
        """
        Initializes a dictionary of the actions for the problem with the
        given values.  It is a dictionary of dictionaries. Positive transfers
        are from state 0 to state 1.

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
                adict[the_key] = {} # start the dictionary
                for trans in range(-j,i+1): # limit transfers to cars on hand
                    if (j + trans) > self.num_cars and trans > 0:
                        continue # don't allow transfers above num_cars
                    elif (i - trans) > self.num_cars and trans < 0:
                        continue # don't allow transfers above num_cars
                    else:
                        adict[the_key][trans] = value
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

        Returns
        -------
        The updated state values in a dictionary.

        """
        delta = 2 * tolerance
        V = self.states.copy() # create a local dictionary of the states
        iter = 0
        while delta > tolerance:
            if verbose:
                print('Iteration: {:d}'.format(iter))
            delta = 0.0 # default to leaving the while loop
            for s,v in V.items():
                # s is the initial state, v is the initial value
                new_V = 0
                action_norm = sum(self.actions[s].values())
                for action,prob in self.actions[s].items():
                    # NOTE: there is only one final state per initial state and
                    # action pair because the actions are deterministic
                    sp = self.take_action(s,action) # final state, s'
                    new_V += self.sum_over_r(sp,V[sp],action) * \
                             (prob/action_norm)
                V[s] = new_V
                delta = max(delta,abs(v - V[s]))
                if verbose:
                    print('State: {:s} :: Value: {:4.3f} :: delta: {:4.3f}'\
                                .format(s,V[s],delta))
            iter += 1
        return V

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

    def take_action(self,initial_state,action):
        """
        Converts an initial state in to a final state using action.  Positive
        action moves cars from state 0 to to state 1.  Reducing the number of
        cars in either state 0 or state 1 to below zero or above 20 is not
        allowed, but this is controled for in the actions parameter.  Ad-hoc
        use of the take_action method may result in unintended consequences.

        Example: initial_state = '1,3'
                 action = -1
                 final_state = '2,2'

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
        state = [x for x in map(int,initial_state.split(','))]
        state[0] -= action
        state[1] += action
        return ','.join(map(str,state))

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

        Example: state = [2,1]
        Returns: ['2,1','1,1','0,1','2,0','1,0','0,0']

        Parameters
        ----------
        state : str
            The state of the two rental lots.

        Returns
        -------
        A list of strings of the ways cars can be rented.
        """
        ways = []
        for im in range(int(state[0])+1):
            for jn in range(int(state[2])+1):
                ways += [str(im)+','+str(jn)]
        return ways
