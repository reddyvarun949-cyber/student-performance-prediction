"""Automated tests for the API. Run from the project root:

    python -m unittest discover -s backend -p "test_*.py" -v

They use Flask's built-in test client, so no server needs to be running.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import app  # noqa: E402

# A typical student
VALID = {
    "study_hours_per_week": 12, "attendance_pct": 80, "previous_sem_gpa": 6.6,
    "assignments_submitted_pct": 80, "internal_test_score": 27, "commute_minutes": 25,
    "sleep_hours": 7, "part_time_job": "No", "family_support": "Medium",
    "internet_access": "Yes", "extracurricular": "No",
}
# A clearly struggling student
STRUGGLING = {
    **VALID, "study_hours_per_week": 4, "attendance_pct": 58, "previous_sem_gpa": 5.0,
    "internal_test_score": 17, "assignments_submitted_pct": 55, "part_time_job": "Yes",
    "commute_minutes": 75, "family_support": "Low",
}


class StudentRiskApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["status"], "ok")

    def test_options(self):
        data = self.client.get("/api/options").get_json()
        self.assertIn("Low", data["categories"]["family_support"])
        self.assertIn("attendance_pct", data["numeric_ranges"])
        self.assertTrue(0 < data["threshold"] < 1)

    def test_valid_prediction(self):
        r = self.client.post("/api/predict", json=VALID)
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertTrue(0 <= body["probability"] <= 1)
        self.assertEqual(body["at_risk"], body["probability"] >= body["threshold"])
        self.assertIn(body["risk_level"], ["Low", "Medium", "High"])
        self.assertIsInstance(body["reasons"], list)
        print("\n  typical student:", body)

    def test_struggling_student_is_flagged_with_reasons(self):
        body = self.client.post("/api/predict", json=STRUGGLING).get_json()
        self.assertTrue(body["at_risk"])
        self.assertEqual(body["risk_level"], "High")
        self.assertGreaterEqual(len(body["reasons"]), 1)
        print("\n  struggling student:", body)

    def test_better_attendance_lowers_risk(self):
        low = self.client.post("/api/predict", json={**STRUGGLING, "attendance_pct": 50}).get_json()
        high = self.client.post("/api/predict", json={**STRUGGLING, "attendance_pct": 95}).get_json()
        self.assertGreater(low["probability"], high["probability"])

    def test_numbers_sent_as_text_are_accepted(self):
        r = self.client.post("/api/predict", json={**VALID, "attendance_pct": "80"})
        self.assertEqual(r.status_code, 200)

    def test_missing_field(self):
        payload = dict(VALID)
        del payload["internal_test_score"]
        r = self.client.post("/api/predict", json=payload)
        self.assertEqual(r.status_code, 422)
        self.assertIn("internal_test_score", r.get_json()["fields"])

    def test_not_a_number(self):
        r = self.client.post("/api/predict", json={**VALID, "sleep_hours": "lots"})
        self.assertEqual(r.status_code, 422)
        self.assertEqual(r.get_json()["fields"]["sleep_hours"], "Must be a number.")

    def test_impossible_attendance(self):
        r = self.client.post("/api/predict", json={**VALID, "attendance_pct": 120})
        self.assertEqual(r.status_code, 422)
        self.assertIn("between", r.get_json()["fields"]["attendance_pct"])

    def test_unknown_category(self):
        r = self.client.post("/api/predict", json={**VALID, "family_support": "Very High"})
        self.assertEqual(r.status_code, 422)
        self.assertIn("Unknown value", r.get_json()["fields"]["family_support"])

    def test_several_errors_reported_together(self):
        r = self.client.post("/api/predict", json={**VALID, "attendance_pct": -3, "internet_access": ""})
        self.assertEqual(r.status_code, 422)
        self.assertEqual(set(r.get_json()["fields"]), {"attendance_pct", "internet_access"})

    def test_body_not_json(self):
        r = self.client.post("/api/predict", data="hello", content_type="text/plain")
        self.assertEqual(r.status_code, 400)

    def test_json_list_is_rejected(self):
        r = self.client.post("/api/predict", json=[1, 2, 3])
        self.assertEqual(r.status_code, 422)

    def test_cors_headers_present(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.headers["Access-Control-Allow-Origin"], "http://localhost:5173")
        pre = self.client.options("/api/predict")
        self.assertEqual(pre.status_code, 204)


if __name__ == "__main__":
    unittest.main()
