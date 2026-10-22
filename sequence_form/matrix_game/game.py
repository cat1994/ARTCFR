import numpy as np
from .simplex import SimplexDomain

class MatrixGame:
    """
    represents the saddle-point problem:
    min_{x\in\Delta} max_{y\in\Delta} x'Ay

    i.e., x, the first player, is the minimizer
    """
    def __init__(self, name, A):
        self._name    = name
        self._A       = A
        self._domains = (SimplexDomain(A.shape[0]), SimplexDomain(A.shape[1]))

    def domain(self, player):
        return self._domains[player]

    def profile_epsilon(self, x, y, regularizer_x=0, regularizer_y=0):
        value_x = self.profile_value(x, y, regularizer_x)
        value_y = self.profile_value_y(y, x, regularizer_y)
        br_x, _ = self.domain(0).support(self.utility_for(0, y, regularizer_x))
        br_y, _ = self.domain(1).support(self.utility_for(1, x, regularizer_y))
        # in no regularization formula, value_x=-value_y, so br_x-value_x + br_y-value_y = br_x+br_y.
        return br_x-value_x + br_y-value_y, br_x - value_x, br_y - value_y, value_x, value_y

    def profile_value(self, x, y, regularizer=0):
        return np.dot(x, self.utility_for(0, y, regularizer))

    def profile_value_y(self, y, x, regularizer=0):
        return np.dot(y, self.utility_for(1, x, regularizer))

    def utility_for(self, player, opponent_strategy, regularizer=0):
        # player's payoff, the player need max!
        # A is the player 0's loss matrix.
        if player == 0:
            return -np.dot(self._A, opponent_strategy)-regularizer  # payoff for x, need max, reg's weight is set negative
        assert player == 1
        return np.dot(self._A.T, opponent_strategy)-regularizer # payoff for y

    def reach(self, player, opponent_strategy):
        return 1

    def sum_info_regret(self, player, strategy, opponent_strategy):
        return 0

    def __str__(self):
        return 'MatrixGame(%s, %dx%d)' % (self._name, self._A.shape[0], self._A.shape[1])
