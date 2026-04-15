from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import auto_repl


class AutoReplTests(unittest.TestCase):
    def test_load_patterns_is_case_insensitive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "patterns.txt"
            path.write_text("test\n", encoding="utf-8")
            raw, compiled, pattern_files = auto_repl.load_patterns(path)
            self.assertEqual(raw, ["test"])
            self.assertEqual(pattern_files, [path])
            self.assertIsNotNone(compiled[0].search("TeSt"))

    def test_discover_regular_files_recursive_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "b").mkdir()
            (root / "a" / "c").mkdir(parents=True)
            (root / "b" / "x.txt").write_text("x", encoding="utf-8")
            (root / "a" / "z.txt").write_text("z", encoding="utf-8")
            (root / "a" / "c" / "y.txt").write_text("y", encoding="utf-8")

            files = auto_repl.discover_regular_files(root)
            rels = [str(p.relative_to(root)) for p in files]
            self.assertEqual(rels, ["a/c/y.txt", "a/z.txt", "b/x.txt"])

    def test_build_output_path_with_suffix(self) -> None:
        self.assertEqual(
            auto_repl.build_output_path(Path("/tmp/test.txt"), "_modified"),
            Path("/tmp/test_modified.txt"),
        )
        self.assertEqual(
            auto_repl.build_output_path(Path("/tmp/archive.tar.gz"), "_modified"),
            Path("/tmp/archive_modified.tar.gz"),
        )
        self.assertEqual(
            auto_repl.build_output_path(Path("/tmp/README"), "_modified"),
            Path("/tmp/README_modified"),
        )

    def test_pattern_directory_aggregation_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a").mkdir()
            (root / "b").mkdir()
            (root / "b" / "02.txt").write_text("two\n", encoding="utf-8")
            (root / "a" / "01.txt").write_text("one\n# skip\n", encoding="utf-8")
            raw, _, files = auto_repl.load_patterns(root)
            self.assertEqual(raw, ["one", "two"])
            rels = [str(p.relative_to(root)) for p in files]
            self.assertEqual(rels, ["a/01.txt", "b/02.txt"])

    def test_overlap_longest_wins_then_pattern_order(self) -> None:
        text = "abc"
        _, patterns = auto_repl.load_patterns_from_strings(["ab", "abc"])
        candidates = auto_repl.collect_match_candidates(text, patterns)
        selected = auto_repl.select_non_overlapping_matches(candidates)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].matched_text, "abc")
        self.assertEqual(selected[0].pattern_index, 1)

    def test_selected_matches_never_overlap(self) -> None:
        text = "ababa"
        _, patterns = auto_repl.load_patterns_from_strings(["aba", "bab", "ab", "ba"])
        candidates = auto_repl.collect_match_candidates(text, patterns)
        selected = auto_repl.select_non_overlapping_matches(candidates)

        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                self.assertFalse(auto_repl.overlaps(selected[i], selected[j]))

        spans = [(item.start, item.end) for item in selected]
        self.assertEqual(spans, [(0, 3), (3, 5)])

    def test_selection_equivalence_randomized_against_legacy(self) -> None:
        rng = random.Random(1337)
        for _ in range(200):
            candidate_count = rng.randint(8, 50)
            candidates: list[auto_repl.MatchCandidate] = []
            for _ in range(candidate_count):
                start = rng.randint(0, 80)
                length = rng.randint(1, 12)
                end = start + length
                pattern_index = rng.randint(0, 7)
                matched_text = "x" * length
                candidates.append(
                    auto_repl.MatchCandidate(
                        start=start,
                        end=end,
                        matched_text=matched_text,
                        pattern_index=pattern_index,
                    )
                )

            optimized = auto_repl.select_non_overlapping_matches(candidates)
            legacy = auto_repl._select_non_overlapping_matches_legacy(candidates)
            optimized_view = [(c.start, c.end, c.pattern_index) for c in optimized]
            legacy_view = [(c.start, c.end, c.pattern_index) for c in legacy]
            self.assertEqual(optimized_view, legacy_view)

    def test_selection_tie_same_start_same_length_pattern_order(self) -> None:
        candidates = [
            auto_repl.MatchCandidate(start=2, end=6, matched_text="AAAA", pattern_index=3),
            auto_repl.MatchCandidate(start=2, end=6, matched_text="BBBB", pattern_index=1),
            auto_repl.MatchCandidate(start=6, end=9, matched_text="CCC", pattern_index=2),
        ]
        selected = auto_repl.select_non_overlapping_matches(candidates)
        view = [(c.start, c.end, c.pattern_index) for c in selected]
        self.assertEqual(view, [(2, 6, 1), (6, 9, 2)])

    def test_token_is_deterministic_and_valid_shape(self) -> None:
        token_a = auto_repl.generate_token(0, "TeSt123")
        token_b = auto_repl.generate_token(0, "TeSt123")
        self.assertEqual(token_a, token_b)
        self.assertTrue(token_a[0].isalpha())
        self.assertTrue(token_a.islower())
        self.assertTrue(token_a.isalnum())

    def test_find_latest_map_file_with_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as primary, tempfile.TemporaryDirectory() as fallback:
            primary_path = Path(primary)
            fallback_path = Path(fallback)
            (fallback_path / "auto-repl-20250101000000.json").write_text("{}", encoding="utf-8")
            (fallback_path / "auto-repl-20260101000000.json").write_text("{}", encoding="utf-8")
            latest, active = auto_repl.find_latest_map_file(primary_path, fallback_path)
            self.assertEqual(latest, fallback_path / "auto-repl-20260101000000.json")
            self.assertEqual(active, fallback_path)

    def test_prune_old_map_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            for ts in ("20260101000000", "20260102000000", "20260103000000"):
                (path / f"auto-repl-{ts}.json").write_text("{}", encoding="utf-8")
            deleted = auto_repl.prune_old_map_files(path, 1)
            self.assertEqual(len(deleted), 2)
            remaining = sorted(p.name for p in auto_repl.list_map_files(path))
            self.assertEqual(remaining, ["auto-repl-20260103000000.json"])

    def test_forward_reverse_roundtrip_single_file_inplace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.txt"
            patterns_path = root / "patterns.txt"
            json_dir = root / "maps"

            original = "test Test TEST TeSt test42 TEST43"
            input_path.write_text(original, encoding="utf-8")
            patterns_path.write_text("test\ntest[0-9]+\n", encoding="utf-8")

            rc = auto_repl.main(
                [
                    "--mode",
                    "forward",
                    "--input",
                    str(input_path),
                    "--patterns",
                    str(patterns_path),
                    "--json-dir",
                    str(json_dir),
                ]
            )
            self.assertEqual(rc, 0)
            transformed = input_path.read_text(encoding="utf-8")
            self.assertNotEqual(transformed, original)

            map_files = auto_repl.list_map_files(json_dir)
            self.assertEqual(len(map_files), 1)
            payload = json.loads(map_files[0].read_text(encoding="utf-8"))
            token_map = payload["token_to_original"]
            self.assertEqual(len(set(token_map.values())), len(token_map))

            rc = auto_repl.main(
                [
                    "--mode",
                    "reverse",
                    "--input",
                    str(input_path),
                    "--json-dir",
                    str(json_dir),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertEqual(input_path.read_text(encoding="utf-8"), original)

    def test_forward_reverse_directory_with_suffix_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "inputs"
            input_dir.mkdir()
            (input_dir / "one.txt").write_text("test TEST", encoding="utf-8")
            (input_dir / "nested").mkdir()
            (input_dir / "nested" / "two.txt").write_text("test42", encoding="utf-8")

            patterns_dir = root / "patterns"
            patterns_dir.mkdir()
            (patterns_dir / "a.txt").write_text("test\ntest[0-9]+\n", encoding="utf-8")
            json_dir = root / "maps"

            rc = auto_repl.main(
                [
                    "--mode",
                    "forward",
                    "--input",
                    str(input_dir),
                    "--patterns",
                    str(patterns_dir),
                    "--output",
                    "_modified",
                    "--json-dir",
                    str(json_dir),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertEqual((input_dir / "one.txt").read_text(encoding="utf-8"), "test TEST")
            self.assertEqual((input_dir / "nested" / "two.txt").read_text(encoding="utf-8"), "test42")

            one_modified = input_dir / "one_modified.txt"
            two_modified = input_dir / "nested" / "two_modified.txt"
            self.assertTrue(one_modified.exists())
            self.assertTrue(two_modified.exists())
            self.assertNotEqual(one_modified.read_text(encoding="utf-8"), "test TEST")
            self.assertNotEqual(two_modified.read_text(encoding="utf-8"), "test42")

            map_files = auto_repl.list_map_files(json_dir)
            self.assertEqual(len(map_files), 1)
            payload = json.loads(map_files[0].read_text(encoding="utf-8"))
            self.assertEqual(len(payload["processed_files"]), 2)
            self.assertGreaterEqual(len(payload["patterns_files"]), 1)

            rc = auto_repl.main(
                [
                    "--mode",
                    "reverse",
                    "--input",
                    str(input_dir),
                    "--output",
                    "_modified",
                    "--json-dir",
                    str(json_dir),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertEqual(one_modified.read_text(encoding="utf-8"), "test TEST")
            self.assertEqual(two_modified.read_text(encoding="utf-8"), "test42")

    def test_reverse_keep_json_uses_map_file_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_dir = root / "maps"
            json_dir.mkdir(parents=True, exist_ok=True)
            input_path = root / "input.txt"
            input_path.write_text("atoken", encoding="utf-8")

            for ts in ("20260101000000", "20260102000000", "20260103000000"):
                payload = {"token_to_original": {"atoken": "value"}}
                (json_dir / f"auto-repl-{ts}.json").write_text(json.dumps(payload), encoding="utf-8")

            rc = auto_repl.main(
                [
                    "--mode",
                    "reverse",
                    "--input",
                    str(input_path),
                    "--map-file",
                    str(json_dir / "auto-repl-20260102000000.json"),
                    "--keep-json",
                    "1",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertEqual(input_path.read_text(encoding="utf-8"), "value")
            self.assertEqual(len(auto_repl.list_map_files(json_dir)), 1)

    def test_reverse_latest_uses_json_dir_then_tmp_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as fallback:
            root = Path(tmp)
            json_dir = root / "maps"
            json_dir.mkdir(parents=True, exist_ok=True)
            input_path = root / "input.txt"
            input_path.write_text("atoken", encoding="utf-8")

            payload = {"token_to_original": {"atoken": "from_json_dir"}}
            (json_dir / "auto-repl-20270101000000.json").write_text(json.dumps(payload), encoding="utf-8")
            (Path(fallback) / "auto-repl-20280101000000.json").write_text(
                json.dumps({"token_to_original": {"atoken": "from_tmp"}}),
                encoding="utf-8",
            )

            with patch.object(auto_repl, "DEFAULT_JSON_DIR", Path(fallback)):
                rc = auto_repl.main(
                    [
                        "--mode",
                        "reverse",
                        "--input",
                        str(input_path),
                        "--json-dir",
                        str(json_dir),
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertEqual(input_path.read_text(encoding="utf-8"), "from_json_dir")


if __name__ == "__main__":
    unittest.main()
