"""G3 unit tests: observation-only payloads, deterministic corruption operators."""
import unittest

from hajj_study.corrupt import (
    DOMAINS, FIELD_DOMAIN, alternative, corrupt_case, source_evidence, stress_influence,
)
from hajj_study.proxy import risk_label

SAMPLE = {
    "Crowd_Density": "Low", "Temperature": "30", "Fatigue_Level": "Low",
    "Stress_Level": "Low", "Queue_Time_minutes": "0", "Emergency_Event": "No",
    "Distance_Between_People_m": "2.5",
}


class CorruptionTests(unittest.TestCase):
    def test_all_seven_domain_fields_partition_exactly_once(self):
        self.assertEqual({f for fields in DOMAINS.values() for f in fields}, set(FIELD_DOMAIN))
        self.assertEqual(sum(len(fields) for fields in DOMAINS.values()), 7)

    def test_numeric_fields_normalized_and_no_type_side_channel(self):
        clean = source_evidence(SAMPLE)
        for f in ("Temperature", "Queue_Time_minutes", "Distance_Between_People_m"):
            self.assertIsInstance(clean[f], float)
        altered, priv = corrupt_case(SAMPLE, "conflicting", "LOW", True,
                                      {"Stress_Level": False, "Crowd_Density": True})
        for f in priv["altered_fields"]:
            value = altered[f]
            self.assertEqual([v["source"] for v in value], ["reading_a", "reading_b"])
            self.assertEqual(type(value[0]["value"]), type(value[1]["value"]))
        sample_mid = dict(SAMPLE)
        sample_mid["Crowd_Density"] = "Medium"
        sample_mid["Temperature"] = "36"
        sample_mid["Stress_Level"] = "Medium"
        sample_mid["Fatigue_Level"] = "Medium"
        sample_mid["Emergency_Event"] = "Yes"
        self.assertEqual(risk_label(sample_mid), "MEDIUM")
        altered, priv = corrupt_case(sample_mid, "conflicting", "MEDIUM", False,
                                       {"Temperature": False})
        self.assertTrue(all(isinstance(x["value"], float) for x in altered["Temperature"]))

    def test_missing_value_is_null_without_truthed_flag(self):
        o, metadata = corrupt_case(SAMPLE, "missing", "LOW")
        self.assertIsNone(o["Crowd_Density"])
        self.assertNotIn("corruption_metadata", o)
        self.assertNotIn("is_corrupted", o)
        self.assertEqual(metadata["altered_domains"], ["crowd"])

    def test_noisy_values_change_within_bounds(self):
        self.assertEqual(alternative("Temperature", "30"), 36)
        self.assertEqual(alternative("Temperature", "45"), 39)
        self.assertEqual(alternative("Queue_Time_minutes", "0"), 20)
        self.assertEqual(alternative("Queue_Time_minutes", "60"), 40)
        self.assertEqual(alternative("Fatigue_Level", "Low"), "High")
        self.assertEqual(alternative("Distance_Between_People_m", "0.5"), 1.1)
        self.assertEqual(alternative("Distance_Between_People_m", "2.5"), 1.9)

    def test_conflicting_clean_position_can_swap(self):
        a, _ = corrupt_case(SAMPLE, "conflicting", "LOW", False,
                            {"Stress_Level": True})
        b, _ = corrupt_case(SAMPLE, "conflicting", "LOW", False,
                            {"Stress_Level": False})
        self.assertEqual(a["Stress_Level"][0]["value"], b["Stress_Level"][1]["value"])
        self.assertEqual(a["Stress_Level"][1]["value"], b["Stress_Level"][0]["value"])
        self.assertEqual(a["Stress_Level"][0]["source"], "reading_a")
        self.assertEqual(b["Stress_Level"][0]["source"], "reading_a")

    def test_gold_is_not_recomputed_from_corrupted_input(self):
        clean_gold = risk_label(SAMPLE)
        observed, metadata = corrupt_case(SAMPLE, "noisy", "LOW")
        self.assertEqual(clean_gold, "LOW")
        self.assertNotEqual(source_evidence(SAMPLE)["Temperature"], observed["Temperature"])
        self.assertEqual(metadata["kind"], "noisy")


if __name__ == "__main__":
    unittest.main()
