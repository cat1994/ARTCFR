from itertools import chain
from itertools import product
import numpy as np
from scipy.sparse import lil_matrix
from sequence_form import matrix_game
from sequence_form.extensive_form_game import extensive_form_game as efg
"""
create Kuhn matrix game
A is opponent utility matrix， our loss matrix
"""


def init_matrix(num_ranks=3):
    assert num_ranks >= 3
    alpha = 1.0 / (num_ranks * (num_ranks - 1))

    # p1: b, cbc, cbf
    # p2: bc/cb, bc/cc, bf/cb, bf/cc
    FOLD = [[0, 0, 1, 1], [0, 0, 0, 0], [-1, 0, -1, 0]]
    SHOWDOWN = [[2, 2, 0, 0], [2, 1, 2, 1], [0, 1, 0, 1]]

    M = 3**num_ranks
    N = 4**num_ranks

    A = np.empty((M, N))
    for i in range(M):
        x = [i / 3**k % 3 for k in range(num_ranks)]
        for j in range(N):
            y = [j / 4**k % 4 for k in range(num_ranks)]

            t = 0
            for c1 in range(num_ranks):
                for c2 in range(num_ranks):
                    if c1 < c2:
                        t += FOLD[x[c1]][y[c2]] - SHOWDOWN[x[c1]][y[c2]]
                    elif c1 > c2:
                        t += FOLD[x[c1]][y[c2]] + SHOWDOWN[x[c1]][y[c2]]

            A[i, j] = -alpha * t

    return matrix_game.MatrixGame('Kuhn%d' % num_ranks, A)


"""
create Kuhn EFG
"""


def init_efg(num_ranks=3,
             prox_infoset_weights=False,
             prox_scalar=-1,
             integer=False,
             all_negative=False):
    assert num_ranks >= 3
    if integer:
        alpha = 1
    else:
        alpha = 1.0 / (num_ranks * (num_ranks - 1))

    dimension = (num_ranks * 4 + 1, num_ranks * 4 + 1)

    # P1 info sets: 0 /c/(bet, check), 1 /c//check/bet/(call, fold)     c:chance
    # P1 sequences: bet, check, /check/bet/call, /check/bet/fold
    first_p1 = np.array(range(1, 4 * num_ranks + 1, 2))
    end_p1 = np.array(range(3, 4 * num_ranks + 3, 2))
    parent_p1 = np.array(
        list(chain.from_iterable((0, 2 + 4 * i) for i in range(0, num_ranks))))
    # P2 info sets: 0 /c/bet/(call, fold), 1 /c/check/(bet, check)
    # P2 sequences: /bet/call, /bet/fold, /check/bet, /check/check
    first_p2 = np.array(range(1, 4 * num_ranks + 1, 2))
    end_p2 = np.array(range(3, 4 * num_ranks + 3, 2))
    parent_p2 = np.array(
        list(chain.from_iterable((0, 0) for i in range(0, num_ranks))))

    if integer:
        A = lil_matrix(dimension, dtype=int)
    else:
        A = lil_matrix(dimension)
    reach = (lil_matrix((len(first_p1), dimension[1])), lil_matrix(
        (len(first_p2), dimension[0])))
    for c1, c2 in product(range(0, num_ranks), repeat=2):  # c1，c2 denote the players hand cards
        if c1 == c2:
            continue
        bet_p1, check_p1, call_p1, fold_p1 = np.array([1, 2, 3, 4]) + c1 * 4
        call_p2, fold_p2, bet_p2, check_p2 = np.array([1, 2, 3, 4]) + c2 * 4

        if c1 > c2:
            winner = -alpha  # negative: loss
        if c2 > c1:
            winner = alpha

        if all_negative:
            offset = -3
        A[bet_p1, call_p2] = winner * 2
        A[bet_p1, fold_p2] = -alpha
        A[check_p1, check_p2] = winner * 1
        A[call_p1, bet_p2] = winner * 2
        A[fold_p1, bet_p2] = alpha

        reach[0][c1 * 2, 0] = 1.0 / num_ranks
        reach[0][c1 * 2 + 1, bet_p2] = alpha
        reach[1][c2 * 2, bet_p1] = alpha
        reach[1][c2 * 2 + 1, check_p1] = alpha
    if all_negative:
        return efg.ExtensiveFormGame(
            "Kuhn%d EFG" % num_ranks,
            A,
            (first_p1, first_p2),
            (end_p1, end_p2),
            (parent_p1, parent_p2),
            prox_infoset_weights=prox_infoset_weights,
            prox_scalar=prox_scalar,
            reach=(reach[0].tocsr(), reach[1].tocsr()),
            all_negative=all_negative,
            offset=2 * offset * (num_ranks * (num_ranks - 1)), )
    else:
        return efg.ExtensiveFormGame(
            "Kuhn%d EFG" % num_ranks,
            A,
            (first_p1, first_p2),
            (end_p1, end_p2),
            (parent_p1, parent_p2),
            prox_infoset_weights=prox_infoset_weights,
            prox_scalar=prox_scalar,
            reach=(reach[0].tocsr(), reach[1].tocsr()),
            all_negative=all_negative, )


def efg_3card_nash_equilibrium(alpha):
    assert alpha > 0 and alpha <= 1.0 / 3
    strategy_p1 = np.array([1,
        alpha, 1 - alpha, 0, 1, 0, 1, 1.0 / 3 + alpha, 2.0 / 3 - alpha,
        3 * alpha, 1 - 3 * alpha, 1, 0
    ])
    strategy_p2 = np.array(
        [1, 0, 1, 1.0 / 3, 2.0 / 3, 1.0 / 3, 2.0 / 3, 0, 1, 1, 0, 1, 0])

    return strategy_p1, strategy_p2


if __name__ == '__main__':
    alpha = 0.17590236
    s1=np.array([1.,         0.17590236, 0.82409764, 0.,         1.,         0.,
 1.,         0.50923569, 0.49076431, 0.52770707, 0.47229293, 1.,
 0.])
    s2=np.array([1.,         0.,         1.,         0.33333333, 0.66666667, 0.33333333,
 0.66666667, 0.,         1.,         1.,         0.,         1.,
 0.])
    t1,t2=efg_3card_nash_equilibrium(alpha)
    if t1.all()==s1.all() and t2.all()==s2.all():
        print("Same")
    else:
        print("Dif")
