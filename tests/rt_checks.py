"""Numerical checks for RT conventions, using stdlib unittest."""
import json
import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sequence_form import paper_adaptive_rtcfr as rt


class RTChecks(unittest.TestCase):
    def test_first_round_signed_discount(self):
        rm = rt._LocalRegret(2, True, 2.0, 0.0)
        rm.update(np.array([2.0, 0.0]), 1)
        np.testing.assert_array_equal(rm.regret, [0.5, -0.5])
        np.testing.assert_array_equal(rm.strategy, [1.0, 0.0])

    def test_negative_infinity_clips_at_first_and_later_rounds(self):
        for t in (1, 2, 100):
            rm = rt._LocalRegret(2, True, 2.0, -math.inf)
            rm.update(np.array([2.0, 0.0]), t)
            self.assertEqual(rm.regret[1], 0.0)
            self.assertEqual(rm.regret[0], t ** 2 / (t ** 2 + 1))

    def test_negative_infinity_configuration_is_standard_json(self):
        settings = rt.resolve_paper_profile('kuhn3_rtdcfr', {'beta': '-inf'})
        self.assertEqual(settings['beta'], -math.inf)
        encoded = json.dumps(rt.json_compatible(settings), allow_nan=False)
        self.assertEqual(json.loads(encoded)['beta'], '-inf')
        for invalid in (math.inf, math.nan):
            with self.assertRaises(ValueError):
                rt.resolve_paper_profile('kuhn3_rtdcfr', {'beta': invalid})

    def test_T1_reference_cancels_RT_weight(self):
        profile = 'matrix_seed0_rtdcfr'
        game = rt.build_paper_game(rt.resolve_paper_profile(profile))
        small = rt.build_paper_solver(game, profile, {'rt_weight': 0.001, 'rt_step': 1})
        large = rt.build_paper_solver(game, profile, {'rt_weight': 100.0, 'rt_step': 1})
        for _ in range(10):
            small.iterate(); large.iterate()
            for actual, expected in zip(small.profile(), large.profile()):
                np.testing.assert_array_equal(actual, expected)

    def test_local_RT_does_not_propagate_child_transformation(self):
        rt.build_paper_game(rt.resolve_paper_profile('matrix_seed0_rtdcfr'))
        domain = rt.TreeplexDomain(5, [1, 3], [3, 5], [0, 1], prox_scalar=1)
        settings = rt.resolve_paper_profile('kuhn3_rtdcfr')
        states = []
        for mode in ('transformed', 'local'):
            player = rt._PlayerRegrets(domain, settings)
            player.update(np.zeros(5), np.array([0., 0., 0., 0.8, -0.2]), 1, mode)
            states.append(player.strategy.copy())
        np.testing.assert_array_equal(states[0][1:3], [1., 0.])
        np.testing.assert_array_equal(states[1][1:3], [0.5, 0.5])
        np.testing.assert_array_equal(states[0][3:5], states[1][3:5])

    def test_profiles_keep_variant_and_provenance_separate(self):
        self.assertEqual(len(rt.paper_profiles()), 32)
        historical = rt.resolve_paper_profile('leduc3_adaptive_rtdcfr')
        local = rt.resolve_paper_profile('leduc3_adaptive_rtdcfr', variant='local_rt')
        self.assertEqual(historical['continuation'], 'transformed')
        self.assertEqual(local['continuation'], 'local')
        self.assertEqual(local['schedule'], 'two_threshold')
        self.assertNotIn('replay_scope', local)
        self.assertEqual(local['baseline_replay_scope'], historical['replay_scope'])


if __name__ == '__main__':
    unittest.main()
