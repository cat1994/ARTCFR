from __future__ import print_function
import warnings
import sys
import math
from collections import defaultdict
from functools import partial
from six.moves import zip as izip
from scipy.stats import entropy
import numpy as np


class TreeplexDomain:
    def __init__(self,
                 dimension,
                 begin,
                 end,
                 parent,
                 seq_to_str=None,
                 prox_infoset_weights=False,
                 prox_scalar=1,
                 ):
        if seq_to_str is None:
            seq_to_str = defaultdict()
        self._forward_order = False
        for i in range(0, len(begin)):
            if parent[i] > begin[i]:
                self._forward_order = True
            assert parent[i] != begin[i]
            assert begin[i] <= end[i]
        if self._forward_order:
            for i in range(0, len(begin)):
                assert parent[i] > begin[i]
        self._dimension = dimension
        self._begin = begin
        self._end = end
        self._parent = parent
        if prox_scalar == -1:
            if prox_infoset_weights:
                prox_weight_scalar = 2.0 / np.sqrt(len(begin))
            else:
                prox_weight_scalar = 1.0
        self._info_w = self._weights(infoset_weights=prox_infoset_weights, weight_scalar=prox_scalar)
        self._prox = TreeplexEntropyProx(
            self,
            self._weights(
                infoset_weights=prox_infoset_weights,
                weight_scalar=prox_scalar))
        self._seq_to_str = seq_to_str
        self.seq_w()

    def info_w(self):
        return self._info_w

    def seq_w(self):
        seq_w = np.ones(self._dimension)
        for i in self.infoset_traversal():
            begin = self._begin[i]
            end = self._end[i]
            seq_w[begin:end] *= self._info_w[i]
        return seq_w

    def dimension(self):
        return self._dimension


    def combine(self, y, alpha, x):
        assert self.is_behavioral_form(x)
        assert self.is_behavioral_form(y)
        seq_y = self.sequence_form(y)
        seq_x = self.sequence_form(x)
        return self.behavioral_form((1.0 - alpha) * seq_y + alpha * seq_x)

    def combine_quadratic(self, y, alpha, x):
        assert self.is_behavioral_form(x)
        assert self.is_behavioral_form(y)
        seq_y = self.sequence_form(y)
        seq_x = self.sequence_form(x)
        return self.behavioral_form(
            (alpha - 1) * (2 * alpha - 1) / (alpha + 1) / (2 * alpha + 1) * seq_y + 6 * alpha / (alpha + 1) / (
                    2 * alpha + 1) * seq_x)

    def prox(self):
        return self._prox

    def smooth_br(self):
        return self._prox.smooth_br

    def ds_br(self):
        return self._prox.ds_br

    def center(self):
        # set to 1/|A_I| for each infoset
        center = np.ones(self._dimension)
        for i in range(0, len(self._begin)):
            infoset_size = self._end[i] - self._begin[i]
            center[self._begin[i]:self._end[i]] /= infoset_size
        return center

    def sequence_form_center(self):
        return self.sequence_form(self.center())

    def diameter(self):
        return self._diameter

    def num_information_sets(self):
        return len(self._begin)

    def get_information_set_index_by_seq(self, seq):
        for i in range(0, len(self._begin)):
            if seq >= self._begin[i] and seq < self._end[i]:
                return i

    def get_information_set_index_by_seqs(self, seqs):
        y = []
        for x in seqs:
            z = self.get_information_set_index_by_seq(x)
            if z not in y:
                y.append(z)
        return y

    def get_parent_info_set(self, info_set):
        return self.get_information_set_index_by_seqs([self.information_set_parent_sequence(info_set)])[0]

    def information_set_parent_sequence(self, info_set):
        return self._parent[info_set]

    def information_set_num_sequences(self, info_set):
        return self._end[info_set] - self._begin[info_set]

    def information_set_first_sequence(self, info_set):
        return self._begin[info_set]

    def information_set_last_sequence(self, info_set):
        return self._end[info_set]

    """
    support function: argmax_{x \\in\Delta} g'x
    Returns support vector in behavioral strategy form.
    """

    def support(self, g):
        response = np.zeros(self._dimension)
        response[self.root_sequence()] = 1
        value = 0
        for i in self.infoset_traversal():
            begin = self._begin[i]
            end = self._end[i]
            parent = self._parent[i]
            idx = np.argmax(g[begin:end])
            response[begin + idx] = 1.0
            if parent == self.root_sequence():
                value += g[begin + idx]
            else:
                g[parent] += g[begin + idx]
        return value, response

    def infoset_regrets(self, g, x):
        # for sequence form
        # bottom up to compute the regret for every information set
        response = np.zeros(self._dimension)
        response[self.root_sequence()] = 1

        what_we_got = g.copy()
        what_we_could_have_gotten = g.copy()

        regrets = np.zeros(len(self._begin))
        for i in self.infoset_traversal():
            begin = self._begin[i]
            end = self._end[i]
            parent = self._parent[i]
            idx = np.argmax(what_we_could_have_gotten[begin:end])
            response[begin + idx] = 1.0

            ev_we_got = x[begin:end].dot(what_we_got[begin:end])


            regrets[i] = max(what_we_got[begin:end]) - ev_we_got
            if parent != self.root_sequence():
                what_we_got[parent] += ev_we_got
        return regrets, response

    def sequence_form(self, x):
        seq = np.copy(x)
        for i in self.reverse_infoset_traversal():
            Z = np.sum(seq[self._begin[i]:self._end[i]])
            if Z > 0.0:
                seq[self._begin[i]:self._end[i]] *= seq[self._parent[i]] / Z
        return seq

    def behavioral_form(self, seq):
        x = np.copy(seq)
        for begin, end in zip(self._begin, self._end):
            Z = np.sum(seq[begin:end])
            if Z == 0:
                x[begin:end] = 1.0 / (end - begin)
            else:
                x[begin:end] /= Z
        return x

    # For kroer17 this computes weights that ensure strong convexity
    # modulus 1. In particular, the recursive formulate gives strong
    # convexity modulus 1/M, where M = max_x ||x||_1.
    #
    # In order to get strong convexity modulus 1 we scale by M.
    def _weights(self, infoset_weights=False, weight_scalar=1.0):
        if infoset_weights == 'kroer15':
            seq_weights = np.full(self._dimension, 0, np.int)
        elif infoset_weights == 'all_one':
            self._diameter = len(self._begin) * np.log(2)  # default max_d = 2
            return np.ones(len(self._begin), float)
        elif infoset_weights == 'level':
            level_info = np.ones(len(self._begin))
            level_seq = np.ones(self._dimension)
            for i in self.infoset_traversal():  # up-bottom
                # find the parent
                parent_seq = self._parent[i]
                while parent_seq != 0:
                    level_info[i] += 1
                    # find the parent seq information set
                    j = self.get_information_set_index_by_seq(parent_seq)
                    parent_seq = self._parent[j]
            return level_info

        else:
            seq_weights = np.full(self._dimension, 0, np.int64)
        weights = np.zeros(len(self._begin), np.float64)
        l1_max = np.zeros(self._dimension)
        depth = np.zeros(self._dimension)
        l1_overall = 0
        depth_overall = 0
        max_simplex_dim = 0

        for i in self.infoset_traversal():  # bottom-up
            begin = self._begin[i]
            end = self._end[i]
            l1 = 1 + np.max(l1_max[begin:end])
            local_depth = 1 + np.max(depth[begin:end])
            size = end - begin
            max_simplex_dim = max(max_simplex_dim, end - begin)
            if infoset_weights == 'kroer15':
                weights[i] = 2 ** local_depth * l1
            elif infoset_weights == 'kroer17':
                weights[i] = 2 + 2 * np.max(seq_weights[begin:end])
            elif infoset_weights == 'farina21':
                weights[i] = 1 + 1 * np.max(seq_weights[begin:end])
            elif infoset_weights == 'test24':
                weights[i] = 2 + 1 * np.sum(seq_weights[begin:end])
            elif infoset_weights == 'ren24':
                weights[i] = local_depth
            if self._parent[i] != self.root_sequence():
                if infoset_weights == 'kroer17' or infoset_weights == 'farina21':
                    seq_weights[self._parent[i]] += 1 * weights[i]
                elif infoset_weights == 'test24':
                    seq_weights[self._parent[i]] += 1 * weights[i]
                l1_max[self._parent[i]] += l1
                depth[self._parent[i]] = max(depth[self._parent[i]], local_depth)
            else:
                l1_overall += l1
                depth_overall = max(depth_overall, local_depth)
        self.seq_depth = depth.copy()
        if infoset_weights == 'kroer15':
            self._diameter = len(self._begin) * depth_overall * 2 ** (
                depth_overall) * l1_overall * np.log(max_simplex_dim)
            return weights
        elif infoset_weights == 'kroer17':
            self._diameter = l1_overall ** 2 * 2 ** (
                depth_overall) * np.log(max_simplex_dim)
            return weights
        elif infoset_weights == 'farina21':
            self._diameter = l1_overall ** 2 * np.log(max_simplex_dim)
            return weights
        elif infoset_weights == 'test24':
            self._diameter = l1_overall ** 2 * np.log(max_simplex_dim)
            return weights
        elif infoset_weights == 'ren24':
            return weights

        else:
            return np.ones(len(self._begin)) * weight_scalar

    def print_sequence_form_constraints(self, f=sys.stdout):
        print(self.num_information_sets(), file=f)
        row = np.zeros(self.dimension(), dtype=int)
        row[self.root_sequence()] = 1
        np.savetxt(f, np.matrix(row), fmt="%i")
        for i in self.infoset_traversal():
            begin = self._begin[i]
            end = self._end[i]
            parent = self._parent[i]
            row = np.zeros(self.dimension(), dtype=int)
            row[parent] = -1
            row[begin:end] = 1
            np.savetxt(f, np.matrix(row), fmt="%i")

    def is_behavioral_form(self, x):
        for i in self.infoset_traversal():
            begin = self._begin[i]
            end = self._end[i]
            if abs(sum(x[begin:end]) - 1) > 1e-8:
                return False
        return True

    def infoset_traversal(self):
        if self._forward_order:
            return range(len(self._begin))
        return range(len(self._begin) - 1, -1, -1)

    def reverse_infoset_traversal(self):
        if self._forward_order:
            return range(len(self._begin) - 1, -1, -1)
        return range(len(self._begin))

    def root_sequence(self):
        if self._forward_order:
            return self._dimension - 1
        else:
            return 0

    def update_v(self, v):
        delta = np.zeros(len(self._begin))
        for i in self.infoset_traversal():
            begin = self._begin[i]
            end = self._end[i]
            delta[i] = np.sum(v[begin:end] ** 2)
        return delta

    @staticmethod
    def project_simplex_with_lower_bounds(x, epsilon):
        """
        Project x onto the simplex with lower bounds epsilon in Euclidean norm.

        The result satisfies y >= epsilon and sum(y) = 1. The entries of
        epsilon must be nonnegative and sum to at most one.
        """
        if not isinstance(x, np.ndarray) or not isinstance(epsilon, np.ndarray):
            raise ValueError("The input and lower bounds must be NumPy arrays.")

        if x.shape != epsilon.shape:
            raise ValueError("The input and lower bounds must have the same shape.")

        if np.any(epsilon < 0):
            raise ValueError("All lower bounds must be nonnegative.")

        if np.sum(epsilon) > 1:
            raise ValueError("The lower bounds must sum to at most one.")

        # Shift the lower bounds to zero.
        z = x - epsilon
        shifted_sum = 1 - np.sum(epsilon)

        if shifted_sum == 0:
            return epsilon.copy()

        if shifted_sum < 0:
            raise ValueError("The lower bounds cannot satisfy sum(y) = 1 and y >= epsilon.")

        # Project the shifted vector onto a simplex with the remaining mass.
        def project_to_simplex(v, s=shifted_sum):
            """Project v onto the simplex {z | z >= 0, sum(z) = s}."""
            n = len(v)
            # Sort v in descending order
            u = np.sort(v)[::-1]
            cssv = np.cumsum(u)
            rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0]
            if len(rho) == 0:
                theta = 0
            else:
                rho = rho[-1]
                theta = (cssv[rho] - s) / (rho + 1.0)
            w = np.maximum(v - theta, 0)
            return w

        z_projected = project_to_simplex(z, shifted_sum)

        # Restore the lower bounds.
        y = z_projected + epsilon

        return y

    def get_seq_epsilon(self, epsilon):
        seq_epsilon = np.full(self.dimension(), epsilon)
        depth_seq = np.zeros(self.dimension())

        for i in self.reverse_infoset_traversal():
            begin = self._begin[i]
            end = self._end[i]
            parent = self._parent[i]
            depth_seq[begin:end] = depth_seq[parent] + 1

        return seq_epsilon ** depth_seq

    def get_perturbed_strategy(self, x, epsilon):
        y = x.copy()
        for i in self.reverse_infoset_traversal():
            begin = self._begin[i]
            end = self._end[i]
            parent = self._parent[i]
            y[begin:end] = self.project_simplex_with_lower_bounds(y[begin:end], np.full(end-begin, epsilon))  #seq_bound[begin:end]/y[parent]
        self.is_behavioral_form(y)
        return y

    def __repr__(self):
        return 'TreeplexDomain(%d)' % self._dimension


class TreeplexEntropyProx:
    def __init__(self, treeplex, weights):
        self._treeplex = treeplex
        self._dimension = treeplex._dimension
        self._weights = weights
        self._begin = treeplex._begin
        self._end = treeplex._end
        self._parent = treeplex._parent
        self._epsilon = 0
        self._center_shift = self.distance_generating_function(self.center())

    def set_epsilon(self, epsilon):
        self._epsilon = epsilon

    def distance_generating_function(self, x):
        assert self._treeplex.is_behavioral_form(x)
        v_vec = np.zeros(self._dimension)
        total = 0
        for i in self._treeplex.infoset_traversal():
            begin = self._begin[i]
            end = self._end[i]
            parent = self._parent[i]
            simplex_dimension = end - begin
            dgf_weight = self._weights[i]

            v = dgf_weight * (-entropy(x[begin: end]) + np.log(simplex_dimension)) + \
                np.dot(x[begin: end], v_vec[begin: end])
            if parent != self._treeplex.root_sequence():
                v_vec[parent] += v
            else:
                total += v
        return total

    def bregman_divergence(self, x, x_center):
        assert self._treeplex.is_behavioral_form(x)
        assert self._treeplex.is_behavioral_form(x_center)
        dgf_x = self.distance_generating_function(x)
        dgf_center = self.distance_generating_function(x_center)
        inner_prod = self.gradient(x_center).dot(
            self._treeplex.sequence_form(x) -
            self._treeplex.sequence_form(x_center))
        return dgf_x - dgf_center - inner_prod

    def weights(self):
        return self._weights

    def center(self):
        _, _, arg = self.smooth_br(0.0, np.zeros(self._dimension), 1.0)
        return arg

    """
    argmin_{x in \Delta} alpha*g'x + beta*D(x, y)
    Express the Bregman update using the smooth best response.

    """

    def __call__(self, alpha, g, beta, y):
        assert self._treeplex.is_behavioral_form(y)
        return self.smooth_br(1., alpha * g - self.gradient(y, beta), beta)

    def ds_br(self, alpha, g, beta, gamma, y, z):
        """
        for DS-OptMD, in REG-CFR
        @param alpha: -1 for min problem
        @param g: value: g-\tau\alpha \nabla d(x^{t-1})
        @param beta: \lambda^{t-1}
        @param gamma: \lambda^{t}-\lambda^{t-1}
        @param y: x^{t-1}
        @param z: x^{1}
        @return:
        """
        # alpha=-1 for min problem
        assert self._treeplex.is_behavioral_form(y)
        return self.smooth_br(1, alpha * g - self.gradient(y, beta) - self.gradient(z, gamma - beta), gamma)

    # solves:
    # argmin_{x\in\Delta} alpha*g'x + beta*d(x)
    def smooth_br(self, alpha, g, beta):
        z = np.zeros(self._dimension)
        z[self._treeplex.root_sequence()] = 1.0
        g *= alpha
        smoothed_br_value = g[self._treeplex.root_sequence()]
        for i in self._treeplex.infoset_traversal():
            begin = self._begin[i]
            end = self._end[i]
            size = end - begin
            parent = self._parent[i]
            if isinstance(beta, np.ndarray): # for ds_omwu
                dgf_weight = beta[i]
            else:
                dgf_weight = beta * self._weights[i]

            # NE refinement
            w = 1 - size * self._epsilon
            offset = np.min(g[begin:end])  # e^g-sum(e^g)= e^{g-offset}/sum(e^{g-offset})
            z[begin:end] = np.exp(-(1.0 / dgf_weight) *
                                  (g[begin:end] - offset) * w)
            Z = np.sum(z[begin:end])
            z[begin:end] /= Z  # in simplex


            # # NE refinement
            z[begin:end] = z[begin:end] * w + self._epsilon  # in perturbed simplex

            best_idx = 0
            best_z = 0
            for idx in range(begin, end):  # this can solve 0log0
                if z[idx] > best_z:
                    best_idx = idx
                    best_z = z[idx]
            v = g[best_idx] + dgf_weight * (
                    np.log(best_z) + np.log(end - begin))
            if parent != self._treeplex.root_sequence():
                g[parent] += v
            else:
                smoothed_br_value += v
        assert self._treeplex.is_behavioral_form(z)
        return smoothed_br_value, g, z

    def gradient(self, strategy, mu=1.0):
        gradient = np.zeros(self._dimension)
        for i in self._treeplex.infoset_traversal():
            begin = self._begin[i]
            end = self._end[i]
            parent = self._parent[i]
            if isinstance(mu, np.ndarray):
                mu2 = mu[i]
            else:
                mu2 = mu * self._weights[i]
            for idx in range(begin, end):
                # catch log-of-zero warning and stop it from printing
                if strategy[idx] != 0:
                    gradient[idx] += mu2 * (
                            1.0 + np.log(strategy[idx]))
                else:
                    gradient[idx] += mu2 * (
                            1.0 + np.log(1e-16))
            gradient[parent] -= mu2 * (
                    1.0 - np.log(end - begin))

        return gradient
