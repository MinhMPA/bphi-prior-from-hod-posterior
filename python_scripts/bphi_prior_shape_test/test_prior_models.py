import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.stats import gaussian_kde

from prior_models import PriorModel, freeze_json


class PriorTests(unittest.TestCase):
    def setUp(self):
        self.values = np.linspace(1, 5, 100)**2
        self.model = PriorModel(self.values)

    def test_gaussian_moments(self):
        self.assertEqual(self.model.mean, self.values.mean())
        self.assertEqual(self.model.std, self.values.std(ddof=0))
        self.assertEqual(self.model.mapped("gaussian", .1, 0), self.values.mean())

    def test_exact_empirical_masses(self):
        selectors = (np.arange(100) + .5) / 100
        np.testing.assert_array_equal(self.model.mapped("empirical", selectors, 0), self.values)
        self.assertEqual(self.model.mapped("empirical", 1., 0), self.values[-1])
        with self.assertRaisesRegex(ValueError, "point masses"):
            self.model.pdf("empirical", np.arange(5))

    def test_selector_orders_without_changing_masses(self):
        selectors = (np.arange(100) + .5) / 100
        model = PriorModel(self.values[::-1])
        np.testing.assert_array_equal(model.mapped("empirical", selectors, 0), self.values)

    def test_kde_matches_scipy(self):
        x = np.linspace(-5, 30, 200)
        np.testing.assert_allclose(self.model.pdf("kde", x), gaussian_kde(self.values)(x), rtol=1e-12)

    def test_normalized_continuous_densities(self):
        for kind in ("gaussian", "kde"):
            value, _ = quad(lambda x: self.model.pdf(kind, x), -np.inf, np.inf)
            self.assertAlmostEqual(value, 1., places=8)

    def test_cdfs(self):
        for kind in ("gaussian", "empirical", "kde"):
            np.testing.assert_allclose(self.model.cdf(kind, [-1e5, 1e5]), [0., 1.])
            self.assertTrue(np.all(np.diff(self.model.cdf(kind, np.linspace(-5, 30, 1000))) >= 0))

    def test_kde_generating_map(self):
        self.assertAlmostEqual(self.model.mapped("kde", .005, 2.), self.values[0] + 2 * self.model.bandwidth)

    def test_no_artificial_positive_boundary(self):
        self.assertLess(self.model.mapped("gaussian", .5, -10.), 0.)
        self.assertLess(self.model.mapped("kde", .005, -10.), 0.)

    def test_bandwidth_scaling(self):
        self.assertEqual(PriorModel(self.values, .5).bandwidth, self.model.bandwidth / 2)

    def test_invalid_data(self):
        for values in ([1], [1, 1], [1, float("nan")], [[1, 2], [3, 4]]):
            with self.assertRaises(ValueError):
                PriorModel(values)

    def test_invalid_bandwidth(self):
        for scale in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                PriorModel(self.values, scale)

    def test_no_silent_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "priors.json"
            freeze_json(path, {"values": [1, 2]})
            freeze_json(path, {"values": [1, 2]})
            with self.assertRaisesRegex(RuntimeError, "Frozen settings differ"):
                freeze_json(path, {"values": [3, 4]})
            self.assertEqual(json.loads(path.read_text()), {"values": [1, 2]})

    def test_frozen_tuples_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            freeze_json(path, {"limits": (None, None)})
            freeze_json(path, {"limits": (None, None)})

    def test_nonfinite_json_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                freeze_json(Path(folder) / "settings.json", {"value": float("nan")})


if __name__ == "__main__":
    unittest.main()
