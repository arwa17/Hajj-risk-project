"""G1 acceptance tests: reconstructed encoding and classification boundaries."""
import math
import unittest

from hajj_study.proxy import (
    PROXY_FIELDS,
    WEIGHTS,
    encoded_fields,
    risk_label,
    risk_label_for_score,
    risk_score,
)

SAMPLE = {
    "Crowd_Density": "Medium",
    "Temperature": "30",
    "Fatigue_Level": "Medium",
    "Stress_Level": "Low",
    "Queue_Time_minutes": "0",
    "Emergency_Event": "No",
    "Distance_Between_People_m": "2.5",
}

class ProxyTests(unittest.TestCase):
    def test_weights_and_source_columns(self):
        self.assertEqual(len(PROXY_FIELDS), 7)
        self.assertAlmostEqual(sum(WEIGHTS.values()), 1.0)
        self.assertEqual(tuple(WEIGHTS), PROXY_FIELDS)

    def test_known_scores(self):
        coded = encoded_fields(SAMPLE)
        self.assertEqual(coded["Temperature"], 1)
        self.assertEqual(coded["Queue_Time_minutes"], 1)
        self.assertEqual(coded["Distance_Between_People_m"], 1)
        self.assertEqual(coded["Emergency_Event"], 0)
        self.assertAlmostEqual(risk_score(SAMPLE), 1.30)
        self.assertEqual(risk_label(SAMPLE), "LOW")

    def test_inverse_distance_and_emergency(self):
        changed = {**SAMPLE, "Distance_Between_People_m": "0.5", "Emergency_Event": "Yes"}
        self.assertEqual(encoded_fields(changed)["Distance_Between_People_m"], 3)
        self.assertEqual(encoded_fields(changed)["Emergency_Event"], 3)
        self.assertAlmostEqual(risk_score(changed) - risk_score(SAMPLE), 0.4)

    def test_thresholds_are_not_rounded(self):
        self.assertEqual(risk_label_for_score(math.nextafter(1.8, 0)), "LOW")
        self.assertEqual(risk_label_for_score(1.8), "MEDIUM")
        self.assertEqual(risk_label_for_score(2.4), "MEDIUM")
        self.assertEqual(risk_label_for_score(math.nextafter(2.4, 3)), "HIGH")

    def test_invalid_input_rejected(self):
        for changed in (
            {**SAMPLE, "Crowd_Density": "Critical"},
            {**SAMPLE, "Temperature": "46"},
            {**SAMPLE, "Queue_Time_minutes": "-1"},
            {**SAMPLE, "Distance_Between_People_m": "nan"},
            {**SAMPLE, "Emergency_Event": "Unknown"},
            {key:value for key,value in SAMPLE.items() if key != "Fatigue_Level"},
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                risk_score(changed)

if __name__ == '__main__':
    unittest.main()
