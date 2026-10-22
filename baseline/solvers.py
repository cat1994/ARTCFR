"""Paper-era comparison solvers, with unrelated research branches removed.

The update order and arithmetic follow the authors' sequence-form experiment
source at revision 8b8b45d. See baseline/README.md for the retained conventions.
"""

import numpy as np

from sequence_form.extensive_form_game.treeplex import TreeplexDomain


class _LocalRegret:
    def __init__(self, dimension, predictive):
        self.strategy = np.ones(dimension) / dimension
        self.regret = np.zeros(dimension)
        self.predictive = predictive

    def step(self, utility):
        value = np.dot(self.strategy, utility)
        immediate = utility - value
        self.regret += immediate
        np.maximum(self.regret, 0, out=self.regret)
        candidate = self.regret + immediate if self.predictive else self.regret
        np.maximum(candidate, 0, out=self.strategy)
        total = np.sum(self.strategy)
        if total <= 0:
            self.strategy.fill(1.0)
            total = len(self.strategy)
        self.strategy /= total
        return value


class _PlayerRegret:
    def __init__(self, domain, predictive):
        self.domain = domain
        self.strategy = domain.center()
        self.tree = isinstance(domain, TreeplexDomain)
        if self.tree:
            self.local = [_LocalRegret(domain.information_set_num_sequences(i), predictive)
                          for i in range(domain.num_information_sets())]
        else:
            self.local = [_LocalRegret(domain.dimension(), predictive)]

    def step(self, utility):
        if not self.tree:
            self.local[0].step(utility)
            self.strategy = self.local[0].strategy.copy()
            return
        for info in self.domain.infoset_traversal():
            begin = self.domain.information_set_first_sequence(info)
            end = self.domain.information_set_last_sequence(info)
            parent = self.domain.information_set_parent_sequence(info)
            value = self.local[info].step(utility[begin:end])
            if parent != self.domain.root_sequence():
                utility[parent] += value
            self.strategy[begin:end] = self.local[info].strategy


class RegretBaseline:
    """Alternating RM+/CFR+ and predictive RM+/CFR+, last or quadratic average."""

    def __init__(self, game, predictive=False, average=True):
        self.game = game
        self.players = [_PlayerRegret(game.domain(i), predictive) for i in (0, 1)]
        self.profile = [game.domain(i).center() for i in (0, 1)]
        self.average = average
        self.iteration = 0
        # The historical accumulator includes the initial profile with weight 1.
        self.total_weight = 1.0

    def iterate(self, count=1):
        for _ in range(count):
            self.players[0].step(self.game.utility_for(0, self.players[1].strategy))
            self.players[1].step(self.game.utility_for(1, self.players[0].strategy))
            self.iteration += 1
            if self.average:
                weight = float((self.iteration + 1) ** 2)
                self.total_weight += weight
                for i in (0, 1):
                    self.profile[i] = self.game.domain(i).combine(
                        self.profile[i], weight / self.total_weight, self.players[i].strategy)
            else:
                self.profile = [p.strategy.copy() for p in self.players]

    def epsilon(self):
        return float(self.game.profile_epsilon(*self.profile)[0])


class MirrorBaseline:
    """Historical OMWU/DOMWU, regularized OMWU/DOMWU, and R-NaD loops."""

    def __init__(self, game, method, prox_weight, rt_weight=0.0, rt_step=1,
                 reg_decay_divisor=2.0):
        self.game = game
        self.method = method
        self.prox = [game.domain(i).prox() for i in (0, 1)]
        self.current = [p.center() for p in self.prox]
        self.profile = [x.copy() for x in self.current]
        self.predictor = [x.copy() for x in self.current]
        self.reference = [x.copy() for x in self.current]
        self.prox_weight = float(prox_weight)
        self.rt_weight = float(rt_weight)
        self.rt_step = int(rt_step)
        self.next_reference = self.rt_step
        self.counter = 1
        self.reg_decay_divisor = float(reg_decay_divisor)
        self.regularized_gap = self._regularized_gap() if method == 'reg_omwu' else None

    @staticmethod
    def _floor(x):
        # Preserve the paper-era in-place numerical floor and operation order.
        x[x < 1e-16] = 1e-16
        return x

    def _entropy_gradient(self, player):
        return (np.log(self._floor(self.predictor[player])) + 1) * self.rt_weight

    def _prox_step(self, player, utility, center):
        # Simplex prox returns (value, strategy); treeplex prox additionally
        # returns a sequence vector. The final element is always behavioral.
        return self.prox[player](-1, utility, self.prox_weight, center)[-1]

    def _regularized_gap(self):
        regularizers = [(np.log(self._floor(x)) + 1) * self.rt_weight for x in self.profile]
        return float(self.game.profile_epsilon(*self.profile, *regularizers)[0])

    def iterate(self, count=1):
        for _ in range(count):
            if self.method == 'rnad':
                rt_x = np.log(self._floor(self.reference[0] / self._floor(self.current[0]))) * self.rt_weight
                utility_x = self.game.utility_for(0, self.current[1]) + rt_x
                updated_x = self._prox_step(0, utility_x, self.current[0])
                rt_y = np.log(self._floor(self.reference[1] / self._floor(self.current[1]))) * self.rt_weight
                utility_y = self.game.utility_for(1, self.current[0]) + rt_y
                self.current[1] = self._prox_step(1, utility_y, self.current[1])
                self.current[0] = updated_x.copy()
            else:
                for player, other in [(0, 1), (1, 0)]:
                    regularizer = self._entropy_gradient(player) if self.method == 'reg_omwu' else 0
                    utility = self.game.utility_for(player, self.current[other], regularizer=regularizer)
                    self.current[player] = self._prox_step(player, utility, self.predictor[player])
                for player, other in [(0, 1), (1, 0)]:
                    regularizer = self._entropy_gradient(player) if self.method == 'reg_omwu' else 0
                    utility = self.game.utility_for(player, self.current[other], regularizer=regularizer)
                    self.predictor[player] = self._prox_step(player, utility, self.predictor[player])
            self.profile = [x.copy() for x in self.current]
            self.counter += 1
            if self.method == 'reg_omwu':
                gap = self._regularized_gap()
                if gap < self.regularized_gap / self.reg_decay_divisor:
                    self.regularized_gap = gap
                    self.rt_weight /= 2
            if self.method == 'rnad' and self.counter >= self.next_reference:
                self.next_reference += self.rt_step
                self.reference = [x.copy() for x in self.current]

    def epsilon(self):
        return float(self.game.profile_epsilon(*self.profile)[0])


def build_baseline(game, settings):
    method = settings['method']
    if method in ('rm_plus', 'predictive_rm_plus'):
        return RegretBaseline(game, predictive=method == 'predictive_rm_plus',
                              average=settings['evaluation'] == 'quadratic_average')
    return MirrorBaseline(game, method, settings['prox_weight'], settings.get('rt_weight', 0),
                          settings.get('rt_step', 1), settings.get('reg_decay_divisor', 2))
