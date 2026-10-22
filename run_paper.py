#!/usr/bin/env python3
"""Run a complete paper experiment group, with explicit per-job budgets."""
import argparse
import json
from pathlib import Path
import time

import run_baselines
from sequence_form import paper_adaptive_rtcfr as rt

ROOT = Path(__file__).resolve().parent


def jobs_for(experiment, variant):
    jobs = []
    suites = json.loads((ROOT / 'configs/rt_suites.json').read_text(encoding='utf-8'))['suites']
    if experiment in ('main', 'discount-ablation'):
        for item in suites[experiment]:
            settings = rt.resolve_paper_profile(item['profile'], item.get('overrides'), variant)
            jobs.append(dict(engine='rt', name=item['name'], profile=item['profile'],
                             game=settings['game'], iterations=item['iterations'],
                             overrides=item.get('overrides', {}), settings=settings))
    if experiment in ('main', 'weights'):
        for name, settings in run_baselines.profiles().items():
            if ('_weights_' in name) != (experiment == 'weights'):
                continue
            jobs.append(dict(engine='baseline', name=name, profile=name,
                             game=settings['game'], iterations=settings['iterations'],
                             settings=settings))
    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=('main', 'discount-ablation', 'weights'), required=True)
    parser.add_argument('--game', choices=('matrix_game', 'kuhn', 'leduc', 'goofspiel', 'liars_dice'))
    parser.add_argument('--iterations', type=int, help='Override each selected budget, e.g. 2 for a smoke check.')
    parser.add_argument('--num-output', type=int, default=101)
    parser.add_argument('--variant', choices=('paper', 'local_rt'), default='paper', help='RT continuation variant.')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if (args.iterations is not None and args.iterations < 1) or args.num_output < 2:
        parser.error('iterations must be positive and num-output must be at least 2')
    jobs = jobs_for(args.experiment, args.variant)
    if args.game:
        jobs = [job for job in jobs if job['game'] == args.game]
    if not jobs:
        parser.error('no runs match the experiment and game')
    if args.iterations is not None:
        for job in jobs:
            job['paper_iterations'] = job['iterations']
            job['iterations'] = args.iterations
            if 'iterations' in job['settings']:
                job['settings']['iterations'] = args.iterations
    manifest = dict(experiment=args.experiment, rt_variant=args.variant, jobs=jobs)
    if args.dry_run:
        print(json.dumps(rt.json_compatible(manifest), indent=2, allow_nan=False))
        return 0
    output = args.output_root or ROOT / 'results' / ('paper_%s_%d' % (args.experiment, time.time_ns()))
    if output.exists():
        parser.error('output-root already exists; choose a new directory')
    output.mkdir(parents=True, exist_ok=False)
    manifest['status'] = 'running'
    def save():
        (output / 'experiment.json').write_text(
            json.dumps(rt.json_compatible(manifest), indent=2, allow_nan=False), encoding='utf-8')
    save()
    try:
        for job in jobs:
            child = output / job['engine'] / job['name']
            common = ['--iterations', str(job['iterations']), '--num-output', str(args.num_output),
                      '--output-root', str(child)]
            if job['engine'] == 'rt':
                command = ['--paper-profile', job['profile'], '--variant', args.variant] + common
                for key, value in job['overrides'].items():
                    flag = {'alpha': 'dcfr-alpha', 'beta': 'dcfr-beta'}.get(key, key.replace('_', '-'))
                    command.append('--%s=%s' % (flag, value))
                rt.main(command)
            else:
                run_baselines.main(['--profile', job['profile']] + common)
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
