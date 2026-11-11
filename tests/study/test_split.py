"""G2 split and feature leakage contract tests (data-independent)."""
import collections
import unittest

from hajj_study.feature_contract import (
    ALLOWED_MODEL_FEATURES,
    EXCLUDED_FROM_MODEL,
    CATEGORICAL_FEATURES,
    validate_features,
)
from hajj_study.split_data import split_indices


class SplitTests(unittest.TestCase):
    def test_feature_contract(self):
        self.assertEqual(len(ALLOWED_MODEL_FEATURES), 17)
        self.assertFalse(set(ALLOWED_MODEL_FEATURES) & EXCLUDED_FROM_MODEL)
        validate_features(list(ALLOWED_MODEL_FEATURES) + list(EXCLUDED_FROM_MODEL))
        self.assertTrue(set(CATEGORICAL_FEATURES).issubset(ALLOWED_MODEL_FEATURES))

    def test_missing_or_extra_feature_fails(self):
        with self.assertRaises(ValueError):
            validate_features(list(ALLOWED_MODEL_FEATURES))
        with self.assertRaises(ValueError):
            validate_features(list(ALLOWED_MODEL_FEATURES) + list(EXCLUDED_FROM_MODEL) + ["Risk_Label"])

    def test_split_reproducible_and_disjoint(self):
        labels = ["LOW"] * 600 + ["MEDIUM"] * 1000 + ["HIGH"] * 400
        a = split_indices(labels)
        b = split_indices(labels)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 2000)
        self.assertEqual(set(a), set(range(2000)))
        self.assertEqual(collections.Counter(a.values()), {"train": 1400, "validation": 200, "test": 400})


if __name__ == '__main__':
    unittest.main()
