"""The local page lists the buttons and serves the saved score without training."""

import json
import threading
import unittest
import urllib.request

from foldlab.screen import FILES, PAGE, SAVED_SCORE, make_httpd, saved_score


class ScreenTests(unittest.TestCase):
    def test_page_lists_the_buttons(self):
        for label in (
            "Short toy fold",
            "Long toy fold",
            "Lysozyme pictures",
            "Compare maps",
            "Sickle-cell picture",
            "Transformer score",
            "Train again",
            "Load protein",
            "Protein code",
        ):
            self.assertIn(label, PAGE)
        self.assertIn("Train again is the only button that trains", PAGE)

    def test_saved_score_matches_the_writeup(self):
        score = saved_score()
        self.assertFalse(score["trained"])
        for key, value in SAVED_SCORE.items():
            self.assertEqual(score[key], value)
        self.assertIn("0.4984", score["text"])
        self.assertIn("1.4658", score["text"])
        self.assertIn("0.8679", score["text"])
        self.assertIn("0.6981", score["text"])
        self.assertIn("not a new training run", score["text"])

    def test_only_known_pictures_are_served(self):
        self.assertIn("demo-short.gif", FILES)
        self.assertIn("lysozyme-hp-core.html", FILES)
        self.assertNotIn("../data/hp_labels.csv", FILES)


class ScreenHttpTests(unittest.TestCase):
    def setUp(self):
        self.httpd = make_httpd(0)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def test_score_route_does_not_train(self):
        page = urllib.request.urlopen(f"http://127.0.0.1:{self.port}/").read().decode()
        self.assertIn("Short toy fold", page)
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/run/score",
            data=b"",
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            payload = json.load(response)
        self.assertFalse(payload["trained"])
        self.assertEqual(payload["mae"], SAVED_SCORE["mae"])
        self.assertEqual(payload["baseline_mae"], SAVED_SCORE["baseline_mae"])

    def test_protein_route_rejects_a_name(self):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/run/protein",
            data=b'{"code":"lysozyme"}',
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request)
        self.assertEqual(caught.exception.code, 400)
        body = json.loads(caught.exception.read().decode())
        self.assertIn("1LYZ", body["error"])

    def test_protein_files_stay_inside_the_folder(self):
        for path in ("/proteins/../data/1lyz.pdb", "/proteins/1lyz-a/../../data/1lyz.pdb"):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}")
            self.assertEqual(caught.exception.code, 404)
