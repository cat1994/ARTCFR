#!/usr/bin/env python3
"""Select RT mu/T using a short budget, with an optional full-budget selected run."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from sequence_form import paper_adaptive_rtcfr as rt


def main(argv=None):
    root = Path(__file__).parent
    defaults_file = root / 'configs' / 'rt_tuning.json'
    defaults = json.loads(defaults_file.read_text(encoding='utf-8'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-profile', required=True)
    parser.add_argument('--mu-grid', help='Comma-separated positive RT weights.')
    parser.add_argument('--T-grid', help='Comma-separated positive integer intervals/thresholds.')
    parser.add_argument('--full-iterations', type=int, help='Default: the corresponding paper figure budget.')
    parser.add_argument('--tuning-iterations', type=int, help='Default: ceil(0.1 * full budget).')
    parser.add_argument('--fraction', type=float, default=defaults['fraction'])
    parser.add_argument('--selection', choices=('final', 'best'), default=defaults['selection'],
                        help='Final short-budget gap, or minimum sampled short-budget gap; ties use grid order.')
    parser.add_argument('--variant', choices=('paper', 'local_rt'), default='paper')
    parser.add_argument('--num-output', type=int, default=101)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--run-selected', action='store_true', help='Also restart the winning configuration for the full budget.')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    try:
        mus = [float(v) for v in args.mu_grid.split(',')] if args.mu_grid else defaults['mu']
        thresholds = [int(v) for v in args.T_grid.split(',')] if args.T_grid else defaults['T']
    except ValueError:
        parser.error('mu-grid must contain numbers and T-grid must contain integers')
    if not mus or not all(math.isfinite(mu) and mu > 0 for mu in mus) or not thresholds or min(thresholds) < 1:
        parser.error('mu and T candidates must be positive (mu must also be finite)')
    if not 0 < args.fraction <= 1 or args.num_output < 2:
        parser.error('fraction must lie in (0,1] and num-output must be at least 2')
    base = rt.resolve_paper_profile(args.paper_profile, variant=args.variant)
    budgets = {'matrix_game': 2000, 'kuhn': 400, 'leduc': 10000, 'goofspiel': 2000, 'liars_dice': 100}
    full_budget = args.full_iterations if args.full_iterations is not None else budgets[base['game']]
    tuning_budget = args.tuning_iterations if args.tuning_iterations is not None else max(1, math.ceil(full_budget * args.fraction))
    if full_budget < 1 or tuning_budget < 1:
        parser.error('iteration budgets must be positive')
    candidates = []
    for mu in mus:
        for threshold in thresholds:
            overrides = dict(rt_weight=mu, max_interval=threshold)
            if not base['method'].startswith('adaptive_'):
                overrides['rt_step'] = threshold
            candidates.append(dict(index=len(candidates), mu=mu, T=threshold, overrides=overrides,
                                   settings=rt.resolve_paper_profile(args.paper_profile, overrides, args.variant)))
    points = sorted(set(round(tuning_budget * i / (args.num_output - 1)) for i in range(args.num_output)))
    manifest = dict(profile=args.paper_profile, variant=args.variant, tuning_iterations=tuning_budget,
                    full_iterations=full_budget, sampled_iterations=points, selection=args.selection,
                    selection_rule='Smallest %s short-budget last-strategy Nash gap; exact ties use candidate order.' % args.selection,
                    provenance=defaults['provenance'], candidates=candidates,
                    runtime=dict(python=sys.version, numpy=rt.np.__version__, scipy=rt.scipy.__version__, platform=platform.platform()),
                    sources_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted((root / 'sequence_form').rglob('*.py'))},
                    profiles_sha256=hashlib.sha256(rt.PROFILE_FILE.read_bytes()).hexdigest(),
                    tuning_config_sha256=hashlib.sha256(defaults_file.read_bytes()).hexdigest())
    if args.dry_run:
        print(json.dumps(rt.json_compatible(manifest), indent=2, allow_nan=False))
        return 0
    if args.output_root is None:
        (root / 'results').mkdir(exist_ok=True)
    output = args.output_root or root / 'results' / ('tuning_%s_%d' % (args.paper_profile, time.time_ns()))
    if output.exists():
        parser.error('output-root already exists; choose a new directory')
    output.mkdir(parents=True, exist_ok=False)
    manifest['status'] = 'running'
    def save():
        (output / 'tuning.json').write_text(json.dumps(rt.json_compatible(manifest),indent=2,allow_nan=False),encoding='utf-8')
    save()
    try:
        game = rt.build_paper_game(base)
        for candidate in candidates:
            solver = rt.PaperRTSolver(game, candidate['settings'])
            previous, records = 0, []
            for iteration in points:
                solver.iterate(iteration - previous)
                previous = iteration
                gap = solver.epsilon()
                if not math.isfinite(gap):
                    raise FloatingPointError('Non-finite gap for candidate %d' % candidate['index'])
                records.append(dict(iteration=iteration, nash_gap=gap))
            candidate.update(final_gap=records[-1]['nash_gap'], best_gap=min(v['nash_gap'] for v in records), records=records)
            candidate['score'] = candidate[args.selection + '_gap']
            save()
        winner = min(candidates, key=lambda item: (item['score'], item['index']))
        manifest['selected_candidate'] = winner['index']
        manifest['selected_overrides'] = winner['overrides']
        with (output / 'candidates.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['index','mu','T','final_gap','best_gap','score'])
            writer.writeheader()
            writer.writerows({key: candidate[key] for key in writer.fieldnames} for candidate in candidates)
        manifest['status'] = 'complete'
        save()
        if args.run_selected:
            child_args = ['--paper-profile', args.paper_profile, '--variant', args.variant,
                          '--iterations', str(full_budget), '--num-output', str(args.num_output),
                          '--output-root', str(output / 'selected_full_run')]
            child_args.extend('--%s=%s' % (key.replace('_','-'), value) for key,value in winner['overrides'].items())
            rt.main(child_args)
            manifest['full_run_status'] = 'complete'
            save()
    except BaseException:
        manifest['status'] = 'incomplete'
        save()
        raise
    print(json.dumps(dict(profile=args.paper_profile, selected=winner['overrides'], score=winner['score'],
                          selection=args.selection, tuning_iterations=tuning_budget),indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
