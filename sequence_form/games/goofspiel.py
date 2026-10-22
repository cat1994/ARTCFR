from sequence_form.extensive_form_game import extensive_form_game as efg
from scipy.sparse import lil_matrix
import copy
import numpy as np
from itertools import product


def init_efg(num_ranks=3,
             prox_infoset_weights=False,
             prox_scalar=-1,
             integer=False,
             all_negative=False,
             num_raise_sizes=1,
             max_bets=2):
    assert num_ranks >= 2
    deck_size = num_ranks
    num_action = num_ranks  # 6

    parent = ([], [])
    begin = ([], [])
    end = ([], [])
    payoff = []
    reach = []
    next_s = [1, 1]
    # below only used for outputting a payoff-shifted constant-sum game when
    # all_negative is True
    payoff_shift = 0
    if all_negative:
        payoff_shift = -15
        payoff_p1 = []

    def _show_down(r, c1, c2):
        if c1 > c2:
            return -r, 0
        elif c1 < c2:
            return 0, -r
        else:
            return -r/2, -r/2  # in Noam's paper, the same bid card split the reward

    def _compute_results(r1, r2): # r denote the loss score
        if r1<r2:
            return -1
        elif r1>r2:
            return 1
        else:
            return 0

    dim = 1
    rnd_start = []
    num_root = []
    num_leaf = []
    num_info = 0  # todo: empty set?
    for i in range(deck_size, 0, -1):
        if i == deck_size:
            num_root.append(1)
            num_leaf.append(i ** 3)

            rnd_start.append(1)
            dim += i ** 2
            num_info += deck_size
        else:
            rnd_start.append(dim)
            num_info += num_leaf[-1] * i

            num_root.append(num_leaf[-1])
            num_leaf.append(num_root[-1] * i ** 3)

            dim += num_root[-1] * i ** 2

    if integer:
        payoff_p1_matrix = lil_matrix((dim, dim), dtype=int)
        payoff_p2_matrix = lil_matrix((dim, dim), dtype=int)
    else:
        payoff_p1_matrix = lil_matrix((dim, dim))
        payoff_p2_matrix = lil_matrix((dim, dim))

    reach_matrix = (lil_matrix((num_info, dim)), lil_matrix((num_info, dim)))


    def _chance(rnd):

        p = 1
        for i in range(rnd+1):
            p /= (deck_size - i)
        return p


    info_index = [0, 0]

    def _build(rnd, deck_list, h1, h2, pre_seq1, pre_seq2, r1, r2):
        # rnd: GAME ROUND: 0-deck_size
        if rnd == deck_size:
            return



        num = len(deck_list)
        for i in range(num):  # deal deck card
            begin1 = next_s[0]
            begin[0].append(begin1)
            end[0].append(begin1 + num)
            parent[0].append(pre_seq1)
            next_s[0] += num

            begin2 = next_s[1]
            begin[1].append(begin2)
            end[1].append(begin2 + num)
            parent[1].append(pre_seq2)
            next_s[1] += num

            chance = _chance(rnd)
            reach_matrix[0][len(begin[0]) - 1, pre_seq2] += chance
            reach_matrix[1][len(begin[1]) - 1, pre_seq1] += chance
            # for p1, p2 actions
            for j in range(num):
                for k in range(num):
                    c0 = deck_list[i]
                    c1 = h1[j]
                    c2 = h2[k]
                    # copy pare
                    _deck_list = copy.copy(deck_list)
                    _h1 = copy.copy(h1)
                    _h2 = copy.copy(h2)

                    # set para
                    _deck_list.remove(c0)
                    _h1.remove(c1)
                    _h2.remove(c2)
                    idx1 = begin1 + j
                    idx2 = begin2 + k

                    # payoff matrix
                    r1_c, r2_c = _show_down(c0, c1, c2)

                    if rnd+1 == deck_size:
                        payoff_p1_matrix[idx1, idx2] += (r1-r2+r1_c-r2_c) * chance



                    #  judge need traverse
                    # subtree root index

                    _build(rnd + 1, _deck_list, _h1, _h2, idx1, idx2, r1+r1_c, r2+r2_c)


    _build(0, [i for i in range(1, deck_size + 1)], [i for i in range(1, deck_size + 1)],
           [i for i in range(1, deck_size + 1)],
           0,0,0,0)

    if all_negative:
        return efg.ExtensiveFormGame(
            'Goofspiel-%d' % num_ranks,
            payoff_p1_matrix,
            begin,
            end,
            parent,
            prox_infoset_weights=prox_infoset_weights,
            prox_scalar=prox_scalar,
            reach=reach_matrix,
            B=payoff_p2_matrix,
            offset=2 * payoff_shift * (deck_size * (deck_size - 1) *
                                       (deck_size - 2)))
    else:
        return efg.ExtensiveFormGame(
            'Goofspiel-%d' % num_ranks,
            payoff_p1_matrix,
            begin,
            end,
            parent,
            prox_infoset_weights=prox_infoset_weights,
            prox_scalar=prox_scalar,
            reach=reach_matrix,
            B=None,
        )
