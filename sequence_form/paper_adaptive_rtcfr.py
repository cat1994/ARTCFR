"""ARTCFR paper experiments with explicit recorded parameter profiles.

The transformed-continuation profiles retain the historical arithmetic order.
The local_rt variant implements the local RT recursion and two-threshold
reference schedule described in the camera-ready manuscript.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy

from .extensive_form_game.treeplex import TreeplexDomain

PROFILE_FILE = Path(__file__).resolve().parents[1] / 'configs' / 'rt_profiles.json'
METHODS = ('rtcfr_plus', 'rtdcfr', 'adaptive_rtcfr_plus', 'adaptive_rtdcfr')


def paper_profiles():
    return json.loads(PROFILE_FILE.read_text(encoding='utf-8'))['profiles']


def resolve_paper_profile(profile, overrides=None, variant='paper'):
    """Resolve every active choice; preserve reported versus reconstructed scope."""
    profiles = paper_profiles()
    if profile not in profiles:
        raise ValueError('Unknown paper profile %r; use --list-profiles' % profile)
    result = copy.deepcopy(profiles[profile])
    allowed = {'rt_weight', 'rt_step', 'max_interval', 'alpha', 'beta', 'theta',
               'multiplier_up', 'multiplier_keep', 'multiplier_down', 'continuation', 'schedule'}
    overrides = {} if overrides is None else {k: v for k, v in overrides.items() if v is not None}
    if set(overrides) - allowed:
        raise ValueError('Unknown paper overrides: %s' % sorted(set(overrides) - allowed))
    if variant not in ('paper', 'local_rt'):
        raise ValueError('variant must be paper or local_rt')
    if variant == 'local_rt':
        result.update(continuation='local', schedule='two_threshold', provenance='configured_local_rt')
        result['field_provenance'].update(continuation='configured_local_rt',schedule='configured_local_rt')
        if 'replay_scope' in result:
            result['baseline_replay_scope']=result.pop('replay_scope')
    result.update(overrides)
    for key in ['rt_step', 'max_interval']:
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 1 or int(value) != value:
            raise ValueError('%s must be a positive integer' % key)
        result[key] = int(value)
    for key in ['rt_weight', 'alpha', 'beta', 'theta', 'multiplier_up', 'multiplier_keep', 'multiplier_down']:
        value = result[key]
        if key == 'beta' and value == '-inf':
            value = -math.inf
        negative_clipping = key == 'beta' and value == -math.inf
        if isinstance(value, bool) or not isinstance(value, (int, float)) or (not math.isfinite(value) and not negative_clipping):
            raise ValueError('%s must be finite (beta also accepts -inf for clipping)' % key)
        result[key] = float(value)
    if result['rt_weight'] <= 0 or result['alpha'] <= 1 or not 0 < result['theta'] < 1:
        raise ValueError('paper RT requires mu>0, alpha>1 and 0<theta<1')
    if result['multiplier_up'] <= 1 or result['multiplier_keep'] <= 0 or not 0 < result['multiplier_down'] < 1:
        raise ValueError('invalid adaptive multipliers')
    if result['continuation'] not in ('transformed', 'local'):
        raise ValueError('continuation must be transformed or local')
    if result['schedule'] not in ('historical_timeout', 'two_threshold'):
        raise ValueError('unknown reference schedule')
    result.update(profile=profile, variant=variant)
    result['active_parameters']=['rt_weight','rt_step','continuation']
    if result['method'].endswith('rtdcfr'):
        result['active_parameters']+=['alpha','beta']
    if result['method'].startswith('adaptive_'):
        result['active_parameters']+=['max_interval','theta','multiplier_up','multiplier_keep','multiplier_down','schedule']
    if overrides:
        result.update(provenance='configured_override', overrides=overrides)
        result['field_provenance'].update({k:'configured_override' for k in overrides})
        if 'replay_scope' in result:
            result['baseline_replay_scope']=result.pop('replay_scope')
    return result


class _LocalRegret:
    def __init__(self, dimension, discounted, alpha, beta):
        self.strategy = np.ones(dimension) / dimension
        self.regret = np.zeros(dimension)
        self.discounted, self.alpha, self.beta = discounted, alpha, beta

    def update(self, utility, t):
        value = np.dot(self.strategy, utility)
        if self.discounted:
            # Preserve the original two accumulation operations.
            self.regret += utility
            self.regret -= value
            for mask, exponent in [(self.regret > 0, self.alpha), (self.regret < 0, self.beta)]:
                if exponent == -math.inf:
                    # beta=-inf means clipping, including the first round.
                    factor = 0.0
                else:
                    try:
                        power = t ** exponent
                        factor = power / (power + 1)
                    except OverflowError:
                        factor = 1.0 if exponent > 0 else 0.0
                self.regret[mask] *= factor
        else:
            self.regret += utility - value
            np.maximum(self.regret, 0, out=self.regret)
        np.maximum(self.regret, 0, out=self.strategy)
        total = np.sum(self.strategy)
        if total <= 0:
            self.strategy.fill(1)
            total = len(self.strategy)
        self.strategy /= total
        return value


class _PlayerRegrets:
    def __init__(self, domain, settings):
        self.domain, self.strategy = domain, domain.center()
        self.efg = isinstance(domain, TreeplexDomain)
        sizes = ([domain.information_set_num_sequences(i) for i in range(domain.num_information_sets())]
                 if self.efg else [domain.dimension()])
        self.rms = [_LocalRegret(n, settings['method'].endswith('rtdcfr'), settings['alpha'], settings['beta']) for n in sizes]

    def update(self, utility, rt, t, continuation):
        if not self.efg:
            self.rms[0].update(utility + rt, t)
            self.strategy = self.rms[0].strategy.copy()
            return
        if continuation == 'transformed':
            utility = utility + rt
        for i in self.domain.infoset_traversal():
            begin = self.domain.information_set_first_sequence(i)
            end = self.domain.information_set_last_sequence(i)
            parent = self.domain.information_set_parent_sequence(i)
            if continuation == 'transformed':
                value = self.rms[i].update(utility[begin:end], t)
            else:
                value = np.dot(self.rms[i].strategy, utility[begin:end])
                self.rms[i].update(utility[begin:end] + rt[begin:end], t)
            if parent != self.domain.root_sequence():
                utility[parent] += value
            self.strategy[begin:end] = self.rms[i].strategy


class PaperRTSolver:
    """Unit-step, post-add RM+/DRM with fixed base mu and explicit references."""
    def __init__(self, game, settings):
        self._game, self.settings = game, copy.deepcopy(settings)
        self._rm_x, self._rm_y = [_PlayerRegrets(game.domain(p), settings) for p in (0, 1)]
        self._x, self._y = self._rm_x.strategy.copy(), self._rm_y.strategy.copy()
        self._reference_strategy_x, self._reference_strategy_y = self._x.copy(), self._y.copy()
        self._t, self._interval, self._reference_step = 1, 1, settings['rt_step']
        self._rt_weight, self._rw, self._exp = settings['rt_weight'], 1.0, self.epsilon()
        self._interval_list, self._gradient_computations = [10], 0

    def profile(self):
        return self._x, self._y

    def epsilon(self):
        return float(self._game.profile_epsilon(self._x, self._y)[0])

    def profile_value(self):
        value = self._game.profile_value(self._x, self._y)
        return float(value[0] if isinstance(value, tuple) else value)

    def gradient_computations(self):
        return self._gradient_computations

    def _reference(self, gap, multiplier):
        self._reference_strategy_x, self._reference_strategy_y = self._x.copy(), self._y.copy()
        self._exp, self._rw = gap, multiplier
        self._interval_list.append(self._interval)
        self._interval = 0

    def iterate(self, num_iterations=1):
        if isinstance(num_iterations, bool) or not isinstance(num_iterations, int) or num_iterations < 0:
            raise ValueError('num_iterations must be a nonnegative integer')
        s = self.settings
        for _ in range(num_iterations):
            # Both RT vectors use the round's old own strategy and reference.
            rt_x = (self._reference_strategy_x - self._rm_x.strategy) * (self._rt_weight * self._rw)
            rt_y = (self._reference_strategy_y - self._rm_y.strategy) * (self._rt_weight * self._rw)
            self._rm_x.update(self._game.utility_for(0, self._rm_y.strategy), rt_x, self._t, s['continuation'])
            self._rm_y.update(self._game.utility_for(1, self._rm_x.strategy), rt_y, self._t, s['continuation'])
            self._x, self._y = self._rm_x.strategy.copy(), self._rm_y.strategy.copy()
            self._gradient_computations += 2
            if self._t == self._reference_step:
                if not s['method'].startswith('adaptive_'):
                    self._reference(self.epsilon(), 1.0)
                else:
                    gap = self.epsilon()
                    if gap < self._exp * s['theta']:
                        self._reference(gap, s['multiplier_up'])
                    elif self._interval >= s['max_interval']:
                        if s['schedule'] == 'historical_timeout':
                            multiplier = s['multiplier_keep'] if gap < self._exp else s['multiplier_down']
                            self._reference(gap, multiplier)
                        elif gap < self._exp or self._interval >= 2 * s['max_interval']:
                            self._reference(gap, s['multiplier_keep'])
                        else:
                            self._rw = s['multiplier_down']
                self._reference_step += s['rt_step']
            self._interval += 1
            self._t += 1


def build_paper_solver(game, profile, overrides=None, variant='paper'):
    return PaperRTSolver(game, resolve_paper_profile(profile, overrides, variant))


def build_paper_game(settings):
    # dtype aliases are needed only by the unchanged historical tree builders.
    if 'int' not in np.__dict__:
        np.int = int
    if 'float' not in np.__dict__:
        np.float = float
    if settings['game'] == 'matrix_game':
        from .games.matrix_game import init_nfg
        return init_nfg(seed=settings['seed'], dimension=(10, 10))
    from .games import goofspiel, liars_dice
    from .poker import kuhn, leduc
    builders = {'kuhn': kuhn.init_efg, 'leduc': leduc.init_efg,
                'goofspiel': goofspiel.init_efg, 'liars_dice': liars_dice.init_efg}
    return builders[settings['game']](
        num_ranks=settings['num_ranks'],
        prox_infoset_weights=settings.get('prox_infoset_weights', False),
        prox_scalar=settings.get('prox_scalar', 1.0))


def json_compatible(value):
    """Encode the clipping sentinel as a JSON string, never a nonstandard number."""
    if isinstance(value, float) and value == -math.inf:
        return '-inf'
    if isinstance(value, dict):
        return {key: json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog='For clipping use --dcfr-beta=-inf. See configs/rt_profiles.json for provenance.')
    parser.add_argument('--paper-profile')
    parser.add_argument('--game',choices=('matrix_game','kuhn','leduc','goofspiel','liars_dice'))
    parser.add_argument('--num-ranks',type=int)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--method','--methods',choices=METHODS)
    parser.add_argument('--list-profiles', action='store_true')
    parser.add_argument('--variant', choices=('paper', 'local_rt'), default='paper')
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--num-output', type=int, default=101)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--rt-weight', type=float)
    parser.add_argument('--rt-step', type=int)
    parser.add_argument('--max-interval', type=int)
    parser.add_argument('--dcfr-alpha', type=float)
    parser.add_argument('--dcfr-beta', type=float)
    parser.add_argument('--theta', type=float)
    parser.add_argument('--multiplier-up', type=float)
    parser.add_argument('--multiplier-keep', type=float)
    parser.add_argument('--multiplier-down', type=float)
    args = parser.parse_args(argv)
    if args.list_profiles:
        print('\n'.join(sorted(paper_profiles())))
        return 0
    if args.iterations < 1 or args.num_output < 2:
        parser.error('iterations must be positive and num-output must be >=2')
    if args.paper_profile is None:
        args.game=args.game or 'kuhn';args.method=args.method or 'adaptive_rtcfr_plus'
        args.seed=0 if args.seed is None else args.seed
        ranks=args.num_ranks if args.num_ranks is not None else {'kuhn':3,'leduc':3,'goofspiel':4,'liars_dice':6}.get(args.game)
        args.paper_profile=(f'matrix_seed{args.seed}_{args.method}' if args.game=='matrix_game'
                            else f'{args.game}{ranks}_{args.method}')
    overrides = dict(rt_weight=args.rt_weight, rt_step=args.rt_step, max_interval=args.max_interval,
                     alpha=args.dcfr_alpha, beta=args.dcfr_beta, theta=args.theta,
                     multiplier_up=args.multiplier_up, multiplier_keep=args.multiplier_keep,
                     multiplier_down=args.multiplier_down)
    settings = resolve_paper_profile(args.paper_profile, overrides, args.variant)
    if (args.game is not None and args.game!=settings['game']) or (args.method is not None and args.method!=settings['method']) or (args.num_ranks is not None and args.num_ranks!=settings['num_ranks']) or (args.seed is not None and args.seed!=settings['seed']):
        parser.error('explicit profile conflicts with game, method or num-ranks')
    manifest = dict(settings=settings, runtime=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform()),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    profiles_sha256=hashlib.sha256(PROFILE_FILE.read_bytes()).hexdigest(), iterations=args.iterations,
                    metric='Nash gap: sum of both best-response improvements', strategy='last',
                    source_files_sha256={str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in sorted(Path(__file__).parent.rglob('*.py'))})
    if args.dry_run:
        print(json.dumps(json_compatible(manifest), indent=2, allow_nan=False))
        return 0
    if args.output_root is None:
        (Path(__file__).resolve().parents[1] / 'results').mkdir(exist_ok=True)
        args.output_root=Path(__file__).resolve().parents[1]/'results'/f'{args.paper_profile}_{time.time_ns()}'
    if args.output_root.exists():
        parser.error('output-root already exists; choose a new run directory')
    args.output_root.mkdir(parents=True,exist_ok=False)
    manifest['output_directory_name']=args.output_root.name
    (args.output_root/'run.json').write_text(json.dumps(json_compatible(manifest),indent=2,allow_nan=False),encoding='utf-8')
    game = build_paper_game(settings)
    solver = PaperRTSolver(game, settings)
    points = sorted(set(round(args.iterations * i / (args.num_output - 1)) for i in range(args.num_output)))
    records, previous, start = [], 0, time.perf_counter()
    for t in points:
        solver.iterate(t - previous);previous = t
        records.append(dict(iteration=t, nash_gap=solver.epsilon(), rt_weight=solver._rt_weight,
                            multiplier=solver._rw, elapsed=time.perf_counter() - start))
    manifest['records'] = records
    (args.output_root / 'run.json').write_text(json.dumps(json_compatible(manifest), indent=2, allow_nan=False), encoding='utf-8')
    with (args.output_root / 'trace.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print(json.dumps(dict(profile=args.paper_profile, final=records[-1], provenance=settings['provenance'],output_root=str(args.output_root)), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
