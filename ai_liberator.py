#!/usr/bin/env python3
"""Bidirectional regex/token replacer."""

from __future__ import annotations

import argparse
import bisect
import concurrent.futures
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Iterable

DEFAULT_JSON_DIR = Path("/tmp")
MAP_FILE_RE = re.compile(r"^auto-repl-(\d{14})\.json$")


Candidate = tuple[int, int, int]  # (start, end, pattern_index)
META_CHARS = frozenset(".^$*+?{}[]\\|()")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regex/token bidirectional replacer")
    parser.add_argument("-m", "--mode", choices=("forward", "reverse"), required=True)
    parser.add_argument("-i", "--input", required=True, help="Input file or directory")
    parser.add_argument(
        "-o",
        "--output",
        help="Output suffix (ex: _modified). If omitted, perform in-place atomic replacement.",
    )
    parser.add_argument(
        "-p",
        "--patterns",
        help="Regex patterns file or directory (one pattern per line). Required in forward mode.",
    )
    parser.add_argument(
        "--json-dir",
        default=str(DEFAULT_JSON_DIR),
        help="Directory for correspondence JSON files (default: /tmp).",
    )
    parser.add_argument(
        "--map-file",
        help="Explicit mapping JSON file to use in reverse mode.",
    )
    parser.add_argument(
        "--keep-json",
        type=int,
        help="In reverse mode, keep only X most recent map files in active JSON dir.",
    )
    parser.add_argument(
        "-j"
        "--jobs",
        type=int,
        default=4,
        help="Forward mode only: number of worker processes (default: 4).",
    )
    return parser.parse_args(argv)


def discover_regular_files(path: Path) -> list[Path]:
    if not path.exists():
        raise ValueError(f"Path not found: {path}")
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise ValueError(f"Path is neither file nor directory: {path}")

    files = [item for item in path.rglob("*") if item.is_file()]
    return sorted(files, key=lambda item: str(item.relative_to(path)).lower())


def load_patterns_from_strings(lines: Iterable[str]) -> tuple[list[str], list[re.Pattern[str]]]:
    raw_patterns: list[str] = []
    compiled: list[re.Pattern[str]] = []
    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        raw_patterns.append(stripped)
        try:
            compiled.append(re.compile(stripped, re.IGNORECASE))
        except re.error as exc:
            raise ValueError(f"Invalid regex at line {line_no}: {stripped!r} ({exc})") from exc
    if not raw_patterns:
        raise ValueError("No usable patterns provided")
    return raw_patterns, compiled


def build_literal_hints(patterns: list[str]) -> list[str | None]:
    hints: list[str | None] = []
    for pattern in patterns:
        if any(char in META_CHARS for char in pattern):
            hints.append(None)
            continue
        hints.append(pattern.lower())
    return hints


def load_patterns(pattern_path: Path) -> tuple[list[str], list[re.Pattern[str]], list[Path]]:
    pattern_files = discover_regular_files(pattern_path)
    all_lines: list[str] = []
    for file_path in pattern_files:
        all_lines.extend(file_path.read_text(encoding="utf-8").splitlines())
    try:
        raw_patterns, compiled = load_patterns_from_strings(all_lines)
    except ValueError as exc:
        raise ValueError(f"{exc} (path: {pattern_path})") from exc
    return raw_patterns, compiled, pattern_files


def collect_match_candidates(
    text: str,
    patterns: list[re.Pattern[str]],
    literal_hints: list[str | None] | None = None,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    append = candidates.append
    text_lower = text.lower() if literal_hints else None
    for pattern_index, pattern in enumerate(patterns):
        if literal_hints:
            hint = literal_hints[pattern_index]
            if hint and hint not in text_lower:
                continue
        for match in pattern.finditer(text):
            if match.start() == match.end():
                continue
            append((match.start(), match.end(), pattern_index))
    return candidates


def overlaps(a: Candidate, b: Candidate) -> bool:
    return not (a[1] <= b[0] or a[0] >= b[1])


def _select_non_overlapping_matches_legacy(candidates: list[Candidate]) -> list[Candidate]:
    """Reference implementation kept for equivalence tests."""
    by_priority = sorted(candidates, key=lambda c: (-(c[1] - c[0]), c[2], c[0]))
    selected: list[Candidate] = []
    for candidate in by_priority:
        if any(overlaps(candidate, already) for already in selected):
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda c: c[0])


def select_non_overlapping_matches(candidates: list[Candidate]) -> list[Candidate]:
    """Choose non-overlapping matches with 'longest wins, tie by pattern order'."""
    by_priority = sorted(candidates, key=lambda c: (-(c[1] - c[0]), c[2], c[0]))
    selected_starts: list[int] = []
    selected_ends: list[int] = []
    selected_items: list[Candidate] = []

    for candidate in by_priority:
        start, end, _ = candidate
        idx = bisect.bisect_left(selected_starts, start)

        if idx > 0 and selected_ends[idx - 1] > start:
            continue
        if idx < len(selected_starts) and selected_starts[idx] < end:
            continue

        selected_starts.insert(idx, start)
        selected_ends.insert(idx, end)
        selected_items.insert(idx, candidate)

    return selected_items


def generate_token(pattern_index: int, matched_text: str) -> str:
    digest = hashlib.sha256(f"{pattern_index}\0{matched_text}".encode("utf-8")).hexdigest()
    return f"a{digest}"


def forward_transform(
    text: str,
    patterns: list[re.Pattern[str]],
    raw_patterns: list[str],
    literal_hints: list[str | None] | None = None,
) -> tuple[str, dict[str, str], list[dict[str, object]]]:
    candidates = collect_match_candidates(text, patterns, literal_hints=literal_hints)
    selected = select_non_overlapping_matches(candidates)

    token_to_original: dict[str, str] = {}
    token_metadata: dict[str, dict[str, object]] = {}
    key_to_token: dict[tuple[int, str], str] = {}
    transformed_parts: list[str] = []
    cursor = 0

    for start, end, pattern_index in selected:
        matched_text = text[start:end]
        transformed_parts.append(text[cursor:start])
        key = (pattern_index, matched_text)
        token = key_to_token.get(key)
        if token is None:
            token = generate_token(pattern_index, matched_text)
            prior = token_to_original.get(token)
            if prior is not None and prior != matched_text:
                raise RuntimeError(
                    f"Token collision for {token}: {prior!r} vs {matched_text!r}"
                )
            key_to_token[key] = token
            token_to_original[token] = matched_text
            token_metadata[token] = {
                "token": token,
                "original": matched_text,
                "pattern_index": pattern_index,
                "pattern": raw_patterns[pattern_index],
            }
        transformed_parts.append(token)
        cursor = end

    transformed_parts.append(text[cursor:])
    mappings = sorted(token_metadata.values(), key=lambda item: str(item["token"]))
    return "".join(transformed_parts), token_to_original, mappings


def reverse_transform(text: str, token_to_original: dict[str, str]) -> str:
    if not token_to_original:
        return text
    pattern = re.compile(
        "|".join(re.escape(token) for token in sorted(token_to_original, key=len, reverse=True))
    )
    return pattern.sub(lambda m: token_to_original[m.group(0)], text)


def timestamp_now() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")


def write_text_atomically(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        temp_name = handle.name
    os.replace(temp_name, path)


def build_output_path(input_path: Path, output_suffix: str | None) -> Path:
    if not output_suffix:
        return input_path

    suffixes = input_path.suffixes
    extension = "".join(suffixes)
    if extension:
        base = input_path.name[: -len(extension)]
    else:
        base = input_path.name
    return input_path.with_name(f"{base}{output_suffix}{extension}")


def write_output(input_path: Path, output_suffix: str | None, content: str) -> Path:
    target = build_output_path(input_path, output_suffix)
    if target.resolve() == input_path.resolve():
        write_text_atomically(target, content)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def map_filename_from_timestamp(ts: str) -> str:
    return f"auto-repl-{ts}.json"


def parse_map_timestamp(path: Path) -> str | None:
    match = MAP_FILE_RE.match(path.name)
    return match.group(1) if match else None


def list_map_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    files = []
    for path in directory.iterdir():
        if not path.is_file():
            continue
        if parse_map_timestamp(path):
            files.append(path)
    return files


def find_latest_map_file(primary_dir: Path, fallback_dir: Path = DEFAULT_JSON_DIR) -> tuple[Path | None, Path | None]:
    primary = list_map_files(primary_dir)
    if primary:
        latest = max(primary, key=lambda p: parse_map_timestamp(p) or "")
        return latest, primary_dir

    if primary_dir.resolve() == fallback_dir.resolve():
        return None, None

    fallback = list_map_files(fallback_dir)
    if fallback:
        latest = max(fallback, key=lambda p: parse_map_timestamp(p) or "")
        return latest, fallback_dir
    return None, None


def prune_old_map_files(directory: Path, keep_count: int) -> list[Path]:
    files = sorted(
        list_map_files(directory),
        key=lambda p: parse_map_timestamp(p) or "",
        reverse=True,
    )
    to_delete = files[keep_count:]
    for path in to_delete:
        path.unlink(missing_ok=True)
    return to_delete


def load_map_file(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Invalid map file format: {path}")
    return loaded


def token_map_from_payload(payload: dict[str, object]) -> dict[str, str]:
    token_to_original = payload.get("token_to_original")
    if isinstance(token_to_original, dict):
        return {str(k): str(v) for k, v in token_to_original.items()}

    mappings = payload.get("mappings")
    if not isinstance(mappings, list):
        raise ValueError("Mapping file is missing both 'token_to_original' and valid 'mappings'")

    reconstructed: dict[str, str] = {}
    for item in mappings:
        if not isinstance(item, dict):
            continue
        token = item.get("token")
        original = item.get("original")
        if token is None or original is None:
            continue
        reconstructed[str(token)] = str(original)
    return reconstructed


def _process_forward_files(
    file_paths: list[str],
    output_suffix: str | None,
    raw_patterns: list[str],
    literal_hints: list[str | None],
) -> dict[str, object]:
    compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in raw_patterns]
    token_to_original: dict[str, str] = {}
    mappings_by_token: dict[str, dict[str, object]] = {}
    processed_files: list[dict[str, str]] = []

    for file_path in file_paths:
        input_file = Path(file_path)
        text = input_file.read_text(encoding="utf-8")
        transformed, file_token_map, file_mappings = forward_transform(
            text,
            compiled_patterns,
            raw_patterns,
            literal_hints=literal_hints,
        )
        output_file = write_output(input_file, output_suffix, transformed)
        processed_files.append({"input": str(input_file.resolve()), "output": str(output_file.resolve())})

        for token, original in file_token_map.items():
            existing = token_to_original.get(token)
            if existing is not None and existing != original:
                raise RuntimeError(f"Token collision across files for {token}: {existing!r} vs {original!r}")
            token_to_original[token] = original
        for item in file_mappings:
            mappings_by_token[str(item["token"])] = item

    return {
        "processed_files": processed_files,
        "token_to_original": token_to_original,
        "mappings": mappings_by_token,
    }


def run_forward(args: argparse.Namespace) -> int:
    if not args.patterns:
        raise ValueError("--patterns is required in forward mode")
    if args.output is not None and not args.output:
        raise ValueError("--output suffix cannot be empty")
    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1")

    input_root = Path(args.input)
    output_suffix = args.output
    pattern_root = Path(args.patterns)
    json_dir = Path(args.json_dir)

    input_files = discover_regular_files(input_root)
    raw_patterns, compiled_patterns, pattern_files = load_patterns(pattern_root)
    literal_hints = build_literal_hints(raw_patterns)
    all_token_to_original: dict[str, str] = {}
    all_mappings: dict[str, dict[str, object]] = {}
    processed_files: list[dict[str, str]] = []

    if args.jobs == 1 or len(input_files) <= 1:
        for input_file in input_files:
            text = input_file.read_text(encoding="utf-8")
            transformed, token_to_original, mappings = forward_transform(
                text,
                compiled_patterns,
                raw_patterns,
                literal_hints=literal_hints,
            )
            output_file = write_output(input_file, output_suffix, transformed)
            processed_files.append({"input": str(input_file.resolve()), "output": str(output_file.resolve())})
            for token, original in token_to_original.items():
                existing = all_token_to_original.get(token)
                if existing is not None and existing != original:
                    raise RuntimeError(f"Token collision across files for {token}: {existing!r} vs {original!r}")
                all_token_to_original[token] = original
            for item in mappings:
                all_mappings[str(item["token"])] = item
    else:
        jobs = min(args.jobs, len(input_files))
        chunks: list[list[str]] = [[] for _ in range(jobs)]
        for idx, input_file in enumerate(input_files):
            chunks[idx % jobs].append(str(input_file))

        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = [
                executor.submit(
                    _process_forward_files,
                    chunk,
                    output_suffix,
                    raw_patterns,
                    literal_hints,
                )
                for chunk in chunks
                if chunk
            ]
            worker_results = [future.result() for future in futures]

        for result in worker_results:
            processed_files.extend(result["processed_files"])
            token_to_original = result["token_to_original"]
            mappings = result["mappings"]
            for token, original in token_to_original.items():
                existing = all_token_to_original.get(token)
                if existing is not None and existing != original:
                    raise RuntimeError(f"Token collision across files for {token}: {existing!r} vs {original!r}")
                all_token_to_original[token] = original
            for token, item in mappings.items():
                all_mappings[token] = item

        processed_files.sort(key=lambda item: item["input"])

    ts = timestamp_now()
    json_dir.mkdir(parents=True, exist_ok=True)
    map_path = json_dir / map_filename_from_timestamp(ts)
    payload = {
        "version": 1,
        "timestamp": ts,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_root": str(input_root.resolve()),
        "processed_files": processed_files,
        "patterns_path": str(pattern_root.resolve()),
        "patterns_files": [str(path.resolve()) for path in pattern_files],
        "id_strategy": "deterministic_sha256_prefixed",
        "case_insensitive": True,
        "token_to_original": all_token_to_original,
        "mappings": sorted(all_mappings.values(), key=lambda item: str(item["token"])),
    }
    map_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return 0


def run_reverse(args: argparse.Namespace) -> int:
    if args.keep_json is not None and args.keep_json < 0:
        raise ValueError("--keep-json must be >= 0")
    if args.output is not None and not args.output:
        raise ValueError("--output suffix cannot be empty")
    if args.jobs != 1:
        raise ValueError("--jobs is only supported in forward mode")

    input_root = Path(args.input)
    output_suffix = args.output
    primary_json_dir = Path(args.json_dir)
    input_files = discover_regular_files(input_root)

    if args.map_file:
        map_path = Path(args.map_file)
        if not map_path.exists():
            raise ValueError(f"Map file not found: {map_path}")
        active_json_dir = map_path.parent
    else:
        map_path, active_json_dir = find_latest_map_file(primary_json_dir, DEFAULT_JSON_DIR)
        if map_path is None:
            raise ValueError(f"No map file found in {primary_json_dir} nor fallback {DEFAULT_JSON_DIR}")
        assert active_json_dir is not None

    payload = load_map_file(map_path)
    token_to_original = token_map_from_payload(payload)
    for input_file in input_files:
        text = input_file.read_text(encoding="utf-8")
        restored = reverse_transform(text, token_to_original)
        write_output(input_file, output_suffix, restored)

    if args.keep_json is not None:
        prune_old_map_files(active_json_dir, args.keep_json)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "forward":
        return run_forward(args)
    return run_reverse(args)


if __name__ == "__main__":
    raise SystemExit(main())
