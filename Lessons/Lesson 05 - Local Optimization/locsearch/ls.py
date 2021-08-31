from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import copy
import math
import signal
import sys

class LocalSearcher(object):

    """Performs local search by calling functions to calculate
    objective function value and make moves on a state.  The search terminates when 
    no improvements to the objective function value have made for max_no_improve 
    iterations (default 1000)
    """

    __metaclass__ = abc.ABCMeta

    # defaults
    max_no_improve = 1000
    update_iter = 100
    copy_strategy = 'deepcopy'

    # placeholders
    best_x = None
    best_f = None
    iterations = 0

    def __init__(self, initial_state=None):
        if initial_state is not None:
            self.state = self.copy_state(initial_state)
        else:
            raise ValueError('No inital_state supplied')

    @abc.abstractmethod
    def move(self):
        """Create a state change"""
        pass

    @abc.abstractmethod
    def objective(self):
        """Calculate state's objective function value"""
        pass

    def copy_state(self, state):
        """Returns an exact copy of the provided state
        Implemented according to self.copy_strategy, one of

        * deepcopy : use copy.deepcopy (slow but reliable)
        * slice: use list slices (faster but only works if state is list-like)
        * method: use the state's copy() method
        """
        if self.copy_strategy == 'deepcopy':
            return copy.deepcopy(state)
        elif self.copy_strategy == 'slice':
            return state[:]
        elif self.copy_strategy == 'method':
            return state.copy()
        else:
            raise RuntimeError('No implementation found for ' +
                               'the self.copy_strategy "%s"' %
                               self.copy_strategy)

    def update(self):
        """
        Prints the current best value and iterations every update_iter iterations.
        """

        if self.iterations == 0:
            print("\n Obj Fun Val | Iterations")
        else:
            print(f"{self.best_f:12.2f} | {self.iterations:d}")


    def localsearch(self):
        """Minimizes the objective function value by local search.

        Parameters
        state : an initial arrangement of the system

        Returns
        (state, objective): the best state and objective function value found.
        """
        self.iterations = 0
        num_moves_no_improve = 0

        # set initial state and compute initial objective value
        self.best_x = self.copy_state(self.state)
        self.best_f = self.objective()
        self.update()
        
        # main local search loop
        while num_moves_no_improve < self.max_no_improve:
            num_moves_no_improve += 1
            self.iterations += 1
            curr_state = self.copy_state(self.state)
            self.move() # stores new state with move in self.state
            new_f = self.objective()
            if new_f < self.best_f:
                num_moves_no_improve = 0
                self.best_x = self.copy_state(self.state)
                self.best_f = new_f
            else: # if move not improvement reset state to curr_state
                self.state = self.copy_state(curr_state)
            if( self.iterations % self.update_iter == 0): # output every update_iter iterations
                self.update() # print output

        # output one last time for final iteration
        self.update()
        
        # Return best state and energy
        return self.best_x, self.best_f