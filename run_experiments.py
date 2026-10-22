#!/usr/bin/env python3
"""Run one RT profile, all main RT profiles, or the discount ablation."""
import argparse
import json
from pathlib import Path
import sys
import time
from sequence_form import paper_adaptive_rtcfr as rt


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--suite' not in argv and not any(arg.startswith('--suite=') for arg in argv):
        if '--help' in argv or '-h' in argv:
            print('Batch: --suite main|discount-ablation [--game GAME] [--iterations N] [--output-root NEW_DIRECTORY] [--dry-run]')
        return rt.main(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('main', 'discount-ablation'), required=True)
    parser.add_argument('--game', choices=('matrix_game', 'kuhn', 'leduc', 'goofspiel', 'liars_dice'))
    parser.add_argument('--iterations', type=int, help='Override all selected budgets, e.g. for a smoke run.')
    parser.add_argument('--num-output', type=int, default=101)
    parser.add_argument('--variant', choices=('paper', 'local_rt'), default='paper')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.iterations is not None and args.iterations < 1:
        parser.error('iterations must be positive')
    if args.num_output < 2:
        parser.error('num-output must be at least 2')
    suite_file = Path(__file__).parent / 'configs' / 'rt_suites.json'
    jobs = json.loads(suite_file.read_text(encoding='utf-8'))['suites'][args.suite]
    profiles = rt.paper_profiles()
    if args.game:
        jobs = [job for job in jobs if profiles[job['profile']]['game'] == args.game]
    if not jobs:
        parser.error('no runs match the selected suite and game')
    resolved = []
    for job in jobs:
        job = dict(job)
        if args.iterations is not None:
            job['iterations'] = args.iterations
        job['settings'] = rt.resolve_paper_profile(job['profile'], job.get('overrides'), args.variant)
        resolved.append(job)
    manifest = dict(suite=args.suite, variant=args.variant, runs=resolved,
                    comparison_scope='New executions of selected configurations; historical curve equality is not implied.')
    if args.dry_run:
        print(json.dumps(rt.json_compatible(manifest), indent=2, allow_nan=False))
        return 0
    output = args.output_root or Path(__file__).parent / 'results' / ('%s_%d' % (args.suite, time.time_ns()))
    if output.exists():
        parser.error('output-root already exists; choose a new run directory')
    output.mkdir(parents=True, exist_ok=False)
    manifest['status'] = 'running'
    manifest_file = output / 'suite.json'
    def save():
        manifest_file.write_text(json.dumps(rt.json_compatible(manifest), indent=2, allow_nan=False), encoding='utf-8')
    save()
    try:
        for job in resolved:
            child_args = ['--paper-profile', job['profile'], '--iterations', str(job['iterations']),
                          '--num-output', str(args.num_output), '--variant', args.variant,
                          '--output-root', str(output / job['name'])]
            for key, value in job.get('overrides', {}).items():
                flag = {'alpha': 'dcfr-alpha', 'beta': 'dcfr-beta'}.get(key, key.replace('_', '-'))
                child_args.append('--%s=%s' % (flag, value))
            rt.main(child_args)
            job['status'] = 'complete'
            save()
    except BaseException:
        manifest['status'] = 'incomplete'
        save()
        raise
    manifest['status'] = 'complete'
    save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
