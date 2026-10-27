from __future__ import print_function
import sys
from collections import defaultdict
import numpy as np
from scipy.sparse import isspmatrix_lil, isspmatrix_csr
from .treeplex import TreeplexDomain


class ExtensiveFormGame:
    """
    represents the saddle-point problem:
    min_{x\in\Delta} max_{y\in\Delta} x'Ay

    i.e., x, the first player, is the minimizer

    Expects A and reach to be of type scipy.sparse.lil_matrix or
    scipy.sparse.csr_matrix
    """

    def __init__(self,
                 name,
                 A,
                 first,
                 end,
                 parent,
                 seq_to_str=None,
                 prox_infoset_weights=False,
                 prox_scalar=1,
                 reach=None,
                 all_negative=False,
                 offset=0,
                 B=None):
        if seq_to_str is None:
            seq_to_str = [defaultdict(), defaultdict()]
        self._name = name
        assert isspmatrix_lil(A) or isspmatrix_csr(A)
        if isspmatrix_csr(A):
            self._A = A
        else:
            self._A = A.tocsr()
        if reach is not None:
            if isspmatrix_csr(reach[0]):
                self._reach = reach
            else:
                self._reach = (reach[0].tocsr(), reach[1].tocsr())
        self._A_T = self._A.transpose()
        self._domains = (TreeplexDomain(
            self._A.get_shape()[0],
            first[0],
            end[0],
            parent[0],
            seq_to_str[0],
            prox_infoset_weights=prox_infoset_weights,
            prox_scalar=prox_scalar),
                         TreeplexDomain(
                self._A.get_shape()[1],
                first[1],
                end[1],
                parent[1],
                seq_to_str[1],
                prox_infoset_weights=prox_infoset_weights,
                prox_scalar=prox_scalar))
        self.all_negative = all_negative
        self.offset = offset
        if B is not None:
            if isspmatrix_csr(B):
                self._B = B
            else:
                self._B = B.tocsr()
        else:
            self._B = None

    def get_reach_matrix(self, player):
        results=self._reach[player].toarray()
        return results

    def domain(self, player):
        return self._domains[player]

    def profile_epsilon(self, x, y, regularizer_x=0, regularizer_y=0):
        value_x, value_y = self.profile_value(x, y, regularizer_x, regularizer_y)
        br_x, _ = self.domain(0).support(self.utility_for(0, y, regularizer_x))
        br_y, _ = self.domain(1).support(self.utility_for(1, x, regularizer_y))
        if self._B is not None:
            return br_x - value_x + br_y - value_y, br_x - value_x, br_y - value_y, value_x, value_y
        else:
            return br_x + br_y, br_x - value_x, br_y - value_y, value_x, value_y

    def profile_value(self, x, y, regularizer_x=0, regularizer_y=0):
        seq1 = self.domain(0).sequence_form(x)
        seq2 = self.domain(1).sequence_form(y)
        return np.dot(seq1, self.utility_for(0, y, regularizer_x)), np.dot(seq2, self.utility_for(1, x, regularizer_y))

    def max_infoset_regret(self, x, y, epsilon):
        if epsilon!=0:

            x_perturbed=self.domain(0).get_perturbed_strategy(x, epsilon)
            y_perturbed=self.domain(1).get_perturbed_strategy(y, epsilon)
        else:
            return 0

        return max(
            self.max_player_infoset_regret(0, x, y_perturbed)[0],
            self.max_player_infoset_regret(1, y, x_perturbed)[0])



    def get_parent_infoset(self, player, index):
        return self.domain(player).get_parent_info_set(index)

    def max_player_infoset_regret(self, player, strategy, opponent_strategy):
        g = self.utility_for(player, opponent_strategy)
        regrets, _ = self.domain(player).infoset_regrets(g, strategy)  # get regret for every information set
        reach = self.reach(player, opponent_strategy)  # reach prob: opponent player mul chance, that is q_{-i}*c_0
        if min(reach)==0:

            print(f"player: {player}\n"
              f"reach: {reach}\n")

        regrets /= reach
        return max(regrets), np.argmax(regrets)
    def sum_info_regret(self, player, strategy, opponent_strategy):
        g = self.utility_for(player, opponent_strategy)
        regrets, _ = self.domain(player).infoset_regrets(g, strategy)  # get regret for every information set
        reach = self.reach(player, opponent_strategy)
        if min(reach)==0:
            return 999999
        regrets /= reach

        return sum(regrets)

    def get_max_infoset_regret_and_index(self, x, y):
        return self.max_player_infoset_regret(0, x, y), self.max_player_infoset_regret(1, y, x)


    def get_infor_index(self, player, low_reach_set):
        # get the opponent reach seq for the low reach information set
        y=[] # store infor set index
        for x in low_reach_set: # get x inforset index
            row_start = self._reach[player].indptr[x]
            row_end = self._reach[player].indptr[x + 1]
            nonzero_indices = self._reach[player].indices[row_start:row_end]

            a = nonzero_indices # get reach seq by y
            for b in a: # get y infor set index
                z=self._domains[1-player].get_information_set_index_by_seq(b)
                if z not in y:
                    y.append(z)
        return y

    def get_reach_seq(self, player, info_set):
        # get the opponent reach seq index for the low reach information set
        y=[] # store infor set index
        for x in info_set: # get x inforset index
            row_start = self._reach[player].indptr[x]
            row_end = self._reach[player].indptr[x + 1]
            nonzero_indices = self._reach[player].indices[row_start:row_end]

            y.append(nonzero_indices)
        return y

    def get_seq_info(self, player, seq):
        return self.domain(player).get_information_set_index_by_seq(seq)

    def get_seqs_info(self, player, seqs):
        results=[]

        for seq in seqs:
            if isinstance(seq, np.ndarray):
                results.append([self.get_seq_info(player, i) for i in seq])
            else:
                results.append(self.get_seq_info(player, seq))
        return results

    def sum_of_player_infoset_regret(self, player, strategy,
                                     opponent_strategy):

        g = self.utility_for(player, opponent_strategy)
        regrets = self.domain(player).infoset_regrets(g, strategy)
        return np.sum(regrets * strategy)

    def utility_for(self, player, opponent_strategy, regularizer=0):
        #  A is opponent utility function, so the function returns the gradient which needs to add negative symbol
        seq = self.domain(1 - player).sequence_form(opponent_strategy)
        if player == 0:
            return -self._A.dot(seq) - regularizer # in prox, it will add an alpha=-1, change to min <-(g-reg),x>

        assert player == 1
        # not consider no zero-sum game
        if self._B is not None:
            return -self._B.dot(seq)
        return self._A_T.dot(seq) - regularizer

    def payoff_max_norm(self):
        return max(self._A.max(), -self._A.min())

    def reach(self, player, opponent_strategy):
        # self._reach: the reach prob of chance in every information: |I_i| * |Q_{-i}|
        # seq: the opponent strategy
        if self._reach is None:
            raise ValueError(
                "The reach argument in __init__ has to be set in order to call reach()"
            )
        seq = self.domain(1 - player).sequence_form(opponent_strategy)
        return self._reach[player].dot(seq)


    def print_payoff_matrix(self,
                            f=sys.stdout,
                            negate=False,
                            all_negative=False):
        constant_term = 0
        if all_negative:
            constant_term = np.max(np.absolute(self._A)) + 1
        if negate:
            if self._B is None:
                B = -np.copy(self._A)
            else:
                B = self._B
        else:
            B = np.transpose(self._A).copy()
        if self._B is None:
            B.data -= constant_term
        for row in B.todense():
            np.savetxt(f, row, fmt="%i")

    def print_matrices(self, f=sys.stdout, all_negative=False):
        if all_negative:
            print("CS", self.offset, file=f)
        else:
            print("GS", file=f)
        print(self.domain(0).dimension(), self.domain(1).dimension(), file=f)
        self.print_payoff_matrix(f=f, negate=True, all_negative=all_negative)
        self.print_payoff_matrix(f=f, negate=False, all_negative=all_negative)
        self.domain(0).print_sequence_form_constraints(f=f)
        self.domain(1).print_sequence_form_constraints(f=f)

    def __str__(self):
        return 'ExtensiveFormGame(%s, %dx%d)' % (self._name, self._A.shape[0],
                                                 self._A.shape[1])
