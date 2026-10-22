import numpy as np
from sequence_form.matrix_game import game as nfg


def init_nfg(seed=3,
             dimension=None,
             ):
    if dimension is None:
        dimension = (3, 3)

    np.random.seed(seed)
    payoff_matrix = np.random.uniform(-1, 1, size=dimension)

    return nfg.MatrixGame(
        'Matrix Game-(%d-%d)' % (dimension[0], dimension[1]),
        A=payoff_matrix)
