#!/usr/bin/env python3
"""Run or tune the comparison algorithms in the ART experiments."""

import argparse
import csv
import datetime
import hashlib
import itertools
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import scipy

from baseline.solvers import build_baseline
from sequence_form.paper_adaptive_rtcfr import build_paper_game

ROOT = Path(__file__).resolve().parent


def profiles():
    return json.loads((ROOT / 'baseline/profiles.json').read_text(encoding='utf-8'))['profiles']


def _positive_grid(text, integer=False):
    values = [float(x) for x in text.split(',')]
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError('Grid values must be finite and positive')
    if integer and any(int(x) != x for x in values):
        raise ValueError('Reference intervals must be integers')
    return [int(x) for x in values] if integer else values


def tune(game, settings, iterations, weight_grid, rt_weight_grid=None, rt_step_grid=None):
    """Choose the smallest final gap at the explicit short tuning horizon."""
    if settings['method'] in ('rm_plus', 'predictive_rm_plus'):
        raise ValueError('RM-type baselines have no prox coefficient to tune')
    if rt_weight_grid and settings['method'] not in ('reg_omwu', 'rnad'):
        raise ValueError('rt-weight-grid is used only by Reg-OMWU and R-NaD')
    if rt_step_grid and settings['method'] != 'rnad':
        raise ValueError('rt-step-grid is used only by R-NaD')
    regularizers = rt_weight_grid or [settings.get('rt_weight', 0)]
    intervals = rt_step_grid or [settings.get('rt_step', 1)]
    results = []
    best_settings, best_gap = None, float('inf')
    for weight, mu, interval in itertools.product(weight_grid, regularizers, intervals):
        parameters = dict(prox_weight=float(weight))
        if settings['method'] in ('reg_omwu', 'rnad'):
            parameters['rt_weight'] = float(mu)
        if settings['method'] == 'rnad':
            parameters['rt_step'] = int(interval)
        candidate = dict(settings, **parameters)
        solver = build_baseline(game, candidate)
        try:
            solver.iterate(iterations)
            gap = solver.epsilon()
        except (FloatingPointError, OverflowError, ValueError):
            gap = float('inf')
        results.append(dict(parameters, final_gap=float(gap) if math.isfinite(gap) else None))
        if math.isfinite(gap) and gap < best_gap:
            best_settings, best_gap = candidate, gap
    if best_settings is None:
        raise RuntimeError('All tuning candidates failed')
    best_settings['parameter_source'] = 'new explicit grid search'
    best_settings['reproduction_scope'] = 'new_tuned_run'
    return best_settings, results


def main(argv=None):
    available = profiles()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true', help='list named paper and weight-ablation profiles')
    parser.add_argument('--profile', choices=sorted(available), default='matrix_seed0_omwu')
    parser.add_argument('--iterations', type=int, help='default: the corresponding figure budget')
    parser.add_argument('--num-output', type=int, default=101)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--prox-weight', type=float, help='the eta value passed to the historical prox coefficient')
    parser.add_argument('--rt-weight', type=float)
    parser.add_argument('--rt-step', type=int)
    parser.add_argument('--weights', choices=['all_one', 'kroer15', 'kroer17', 'farina21'])
    parser.add_argument('--tune', action='store_true', help='run an explicit short grid search before the full run')
    parser.add_argument('--tuning-iterations', type=int, help='default: one tenth of the run budget, at least 1')
    parser.add_argument('--weight-grid', help='comma-separated prox coefficients; default: 20 log-spaced values .01 to 10')
    parser.add_argument('--rt-weight-grid', help='optional comma-separated positive RT weights for tuning')
    parser.add_argument('--rt-step-grid', help='optional comma-separated positive integer reference intervals')
    parser.add_argument('--dry-run', action='store_true', help='print resolved settings without constructing a game')
    args = parser.parse_args(argv)
    if args.list:
        print('\n'.join(sorted(available)))
        return 0
    settings = dict(available[args.profile])
    for key in ['prox_weight', 'rt_weight', 'rt_step']:
        value = getattr(args, key)
        if value is not None:
            if not math.isfinite(value) or value <= 0:
                parser.error(key + ' must be finite and positive')
            settings[key] = value
            settings['parameter_source'] = 'explicit command-line override'
            settings['reproduction_scope'] = 'new_parameter_run'
    if args.weights:
        settings['prox_infoset_weights'] = args.weights
        settings['parameter_source'] = 'explicit command-line override'
        settings['reproduction_scope'] = 'new_parameter_run'
    if settings['method'] in ('rm_plus', 'predictive_rm_plus') and any(
            getattr(args, key) is not None for key in ['prox_weight', 'rt_weight', 'rt_step']):
        parser.error('prox-weight, rt-weight, and rt-step are not used by RM-type baselines')
    iterations = settings['iterations'] if args.iterations is None else args.iterations
    tuning_iterations = max(1, iterations // 10) if args.tuning_iterations is None else args.tuning_iterations
    if iterations < 1 or args.num_output < 2 or tuning_iterations < 1:
        parser.error('iterations and tuning-iterations must be positive; num-output must be at least 2')
    settings['iterations'] = iterations
    if args.dry_run:
        print(json.dumps(dict(profile=args.profile, settings=settings), indent=2))
        return 0
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = args.output_root or ROOT / 'results' / ('baseline_' + args.profile + '_' + stamp)
    if output.exists():
        parser.error('output-root already exists; choose a new directory')
    game = build_paper_game(settings)
    tuning = None
    if args.tune:
        try:
            grid = _positive_grid(args.weight_grid) if args.weight_grid else np.logspace(-2, 1, 20)
            mu_grid = _positive_grid(args.rt_weight_grid) if args.rt_weight_grid else None
            interval_grid = _positive_grid(args.rt_step_grid, integer=True) if args.rt_step_grid else None
            settings, tuning = tune(game, settings, tuning_iterations, grid, mu_grid, interval_grid)
        except ValueError as exc:
            parser.error(str(exc))
    solver = build_baseline(game, settings)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(profile=args.profile, settings=settings, status='running',
                    runtime=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                    source_sha256=hashlib.sha256((ROOT / 'baseline/solvers.py').read_bytes()).hexdigest())
    if tuning is not None:
        manifest['tuning'] = dict(iterations=tuning_iterations, selection='final_gap', candidates=tuning)
    record_path = output / 'run.json'
    record_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    points = sorted(set(round(iterations * i / (args.num_output - 1)) for i in range(args.num_output)))
    records, previous, start = [], 0, time.perf_counter()
    try:
        for iteration in points:
            solver.iterate(iteration - previous)
            previous = iteration
            gap = solver.epsilon()
            if not math.isfinite(gap):
                raise FloatingPointError('non-finite exploitability')
            records.append(dict(iteration=iteration, nash_gap=gap, elapsed=time.perf_counter() - start))
        manifest['status'] = 'complete'
    except Exception as exc:
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        manifest['records'] = records
        record_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        if records:
            with (output / 'trace.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
    print(json.dumps(dict(profile=args.profile, final=records[-1], output_root=str(output)), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
