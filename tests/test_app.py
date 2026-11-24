import json
import tempfile
import unittest
from pathlib import Path

from app import create_app, connect_db


class FocusAnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        db_path = Path(self.temp_dir.name) / "test.db"
        self.app = create_app({"TESTING": True, "DB_PATH": str(db_path), "SECRET_KEY": "test"})
        self.client = self.app.test_client()

    def tearDown(self):
        self.temp_dir.cleanup()

    def valid_payload(self):
        return {
            "subject": "Algorithms",
            "goal": "Review graph traversal",
            "planned_minutes": 25,
            "study_minutes": 24.5,
            "distraction_minutes": 2,
            "distraction_count": 1,
            "break_minutes": 5,
            "concentration": 8,
        }

    def test_all_pages_start(self):
        for path in ["/", "/dashboard", "/study", "/analysis", "/recommendations", "/reports", "/history"]:
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertIn(b"Focusly", response.data)
                if path == "/analysis":
                    self.assertIn(b"Hold-out accuracy", response.data)
                    self.assertIn(b"225 test examples", response.data)

    def test_prediction_is_real_and_explains_confidence(self):
        response = self.client.post("/api/predict", json={
            "study_minutes": 45, "distraction_minutes": 4, "break_minutes": 8,
            "concentration": 7, "distraction_count": 2,
        })
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertIn(result["category"], ["Low focus", "Medium focus", "High focus"])
        self.assertGreaterEqual(result["confidence"], 0)
        self.assertLessEqual(result["confidence"], 1)
        self.assertTrue(result["explanation"])
        self.assertTrue(result["recommendations"])

    def test_invalid_values_rejected(self):
        response = self.client.post("/api/predict", json={
            "study_minutes": 45, "distraction_minutes": 4, "break_minutes": 8,
            "concentration": 11, "distraction_count": 2,
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn("1 to 10", response.get_json()["error"])

    def test_session_save_persists_and_returns_prediction(self):
        response = self.client.post("/api/sessions", json=self.valid_payload())
        self.assertEqual(response.status_code, 201, response.get_json())
        result = response.get_json()
        self.assertEqual(result["subject"], "Algorithms")
        self.assertTrue(result["id"])
        self.assertIn("confidence", result)
        history = self.client.get("/history")
        self.assertIn(b"Algorithms", history.data)
        analysis = self.client.get("/analysis")
        self.assertIn(b"In plain language", analysis.data)
        self.assertIn(b"The model leaned on", analysis.data)
        stats = self.client.get("/api/stats").get_json()
        self.assertEqual(stats["sessions"], 1)
        self.assertAlmostEqual(stats["study"], 24.5)

    def test_history_deletion(self):
        created = self.client.post("/api/sessions", json=self.valid_payload()).get_json()
        response = self.client.post(f"/history/{created['id']}/delete", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"No saved sessions yet", response.data)
        self.assertEqual(self.client.get("/api/stats").get_json()["sessions"], 0)

    def test_charts_return_png_for_empty_and_populated_db(self):
        for chart in ["daily", "weekly", "distractions", "focus", "subjects"]:
            response = self.client.get(f"/charts/{chart}.png")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.mimetype, "image/png")
            self.assertTrue(response.data.startswith(b"\x89PNG"))
        self.client.post("/api/sessions", json=self.valid_payload())
        for chart in ["daily", "weekly", "distractions", "focus", "subjects"]:
            response = self.client.get(f"/charts/{chart}.png")
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.data.startswith(b"\x89PNG"))

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")


if __name__ == "__main__":
    unittest.main()
