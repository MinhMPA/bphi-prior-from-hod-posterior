"""Check that the released responses reproduce the fiducial prior measures."""
import csv
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import inference_inputs as inputs


class PublicPriorDataTests(unittest.TestCase):
    def test_fiducial_selection_and_moments(self):
        with patch.dict(os.environ, {}, clear=True):
            table, settings, hashes = inputs.read_prior_inputs()
        config = inputs.read_config(Path(__file__).parent / 'configs/joint_empirical_k09_09_10.json')
        expected = {'LRG1': (2.556053628109501, .4234204391477927),
                    'LRG2': (3.5396246285693183, .4176022869711201),
                    'LRG3': (3.946859868212199, .188427499490963)}
        self.assertEqual(len(table), 300)
        self.assertIn(str(inputs.PUBLIC_PRIORS), hashes)
        for sample, spec in config['tracers'].items():
            prior = inputs.select_prior(table, spec, settings)
            self.assertEqual(len(prior['values']), 100)
            np.testing.assert_allclose([prior['mean'], prior['std']], expected[sample], rtol=0, atol=1e-12)

    def test_assembly_values_are_paired_not_used_as_standard(self):
        with inputs.PUBLIC_PRIORS.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        standard = [r for r in rows if r['sample'] == 'LRG3']
        assembly = [r for r in rows if r['sample'] == 'LRG3_AB']
        self.assertEqual(len(rows), 400)
        self.assertEqual(len(assembly), 100)
        self.assertEqual([(r['draw_id'], r['source_row']) for r in standard],
                         [(r['draw_id'], r['source_row']) for r in assembly])
        self.assertTrue(all(float(r['Bcent']) == -.04 and float(r['Bsat']) == -.17 for r in assembly))
        np.testing.assert_allclose(np.mean([float(r['bphi']) for r in assembly]),
                                   3.94542623, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
