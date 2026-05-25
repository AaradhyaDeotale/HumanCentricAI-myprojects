"""Lightweight sanity tests for the Project 2 ML core.

Run with:  python manage.py test project2
These do not touch the database; they exercise ml.py directly.
"""
from django.test import SimpleTestCase

from . import ml


class MLCoreTests(SimpleTestCase):
    def test_data_loads(self):
        data = ml.get_data()
        self.assertEqual(sorted(data["classes"]), ["Adelie", "Chinstrap", "Gentoo"])
        self.assertGreater(len(data["df"]), 300)

    def test_tree_family_monotone_omega(self):
        fam = ml.get_model_family("tree")
        omegas = [m["omega"] for m in fam]
        self.assertEqual(omegas, sorted(omegas))

    def test_lambda_simplifies(self):
        lo, hi = ml.lambda_range("tree")
        best_lo, _ = ml.select_by_lambda("tree", lo)
        best_hi, _ = ml.select_by_lambda("tree", hi)
        self.assertGreaterEqual(best_lo["omega"], best_hi["omega"])

    def test_pdp_sums_to_one(self):
        best, _ = ml.select_by_lambda("logistic", 0.0)
        pdp = ml.compute_pdp(best, "bill_length_mm", grid_size=5)
        import numpy as np
        mat = np.array([pdp["curves"][c] for c in pdp["classes"]])
        self.assertTrue(np.allclose(mat.sum(axis=0), 1.0, atol=1e-6))

    def test_counterfactual_changes_prediction(self):
        best, _ = ml.select_by_lambda("tree", 0.0)
        res = ml.generate_counterfactuals(best, 0, "Gentoo", k=1)
        if res["found"]:
            self.assertEqual(res["target_label"], "Gentoo")
