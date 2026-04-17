# ai-liberator

Regex-based text tokenization and restoration tool.

This project replaces matched text with deterministic IDs and can restore original text later using generated JSON mapping files.

## Features

- Forward mode:
  - Read regex patterns from a file or a directory (recursive).
  - Process input from a file or a directory (recursive).
  - Replace matches with deterministic IDs (`a` + SHA-256 hex, lowercase alnum, no hyphen).
  - Save one mapping JSON per run: `ai-liberator-map-YYYYMMDDHHMMSS.json`.
- Reverse mode:
  - Restore IDs to original text using a mapping JSON.
  - Auto-pick latest mapping from `--json-dir` (fallback to `/tmp`) when `--map-file` is omitted.
- Output behavior:
  - In-place write by default.
  - Optional suffix output (`--output _modified`), e.g. `test.txt` -> `test_modified.txt`.
  - Optional output directory (`--output-dir`), created automatically when missing.
  - Optional path renaming (`--rename-paths`) for directory names and file stems (extensions preserved).
- Matching behavior:
  - Case-insensitive matching (`re.IGNORECASE`) for all patterns.
  - Overlap resolution: longest match wins, tie-break by pattern order, no overlapping replacements.
- Performance:
  - Optimized overlap selection.
  - Optional pattern literal prefiltering.
  - Optional forward multiprocessing via `--jobs`.
- File handling:
  - Non-UTF-8 input files are skipped with a warning instead of failing the whole run.

## Requirements

- Python 3.10+ (tested with Python 3.x stdlib only)

No external dependencies are required.

## Installation

Clone the repository:

```bash
git clone <your-repo-url>
cd ai-liberator
```

Run directly:

```bash
python3 ai_liberator.py --help
```

## Usage

### Forward mode (replace text -> IDs)

```bash
python3 ai_liberator.py \
  --mode forward \
  --input ./data/input.txt \
  --patterns ./patterns.txt \
  --json-dir /tmp
```

Directory input and directory patterns:

```bash
python3 ai_liberator.py \
  --mode forward \
  --input ./data \
  --patterns ./patterns \
  --json-dir /tmp
```

With output suffix:

```bash
python3 ai_liberator.py \
  --mode forward \
  --input ./data \
  --patterns ./patterns \
  --output _modified \
  --output-dir ./out \
  --json-dir /tmp
```

With multiprocessing:

```bash
python3 ai_liberator.py \
  --mode forward \
  --input ./data \
  --patterns ./patterns \
  --json-dir /tmp \
  --jobs 4
```

### Reverse mode (IDs -> original text)

Auto-pick latest map from `--json-dir` (fallback `/tmp`):

```bash
python3 ai_liberator.py \
  --mode reverse \
  --input ./data \
  --output-dir ./restored \
  --json-dir /tmp
```

Use explicit map file:

```bash
python3 ai_liberator.py \
  --mode reverse \
  --input ./data \
  --map-file /tmp/ai-liberator-map-20260415123456.json
```

Keep only the newest N mapping files after reverse:

```bash
python3 ai_liberator.py \
  --mode reverse \
  --input ./data \
  --json-dir /tmp \
  --keep-json 10
```

## CLI reference

```text
--mode {forward,reverse}   Required.
--input PATH               Required. File or directory.
--patterns PATH            Required in forward mode. File or directory.
--output SUFFIX            Optional output suffix; omitted means in-place.
--output-dir PATH          Optional output directory; auto-created if missing.
-d, --json-dir PATH        Mapping directory (default: /tmp).
--map-file PATH            Explicit mapping file (reverse mode).
--keep-json N              Keep only N newest mapping files (reverse mode).
--jobs N                   Forward mode workers (default: 1).
--rename-paths             Also transform directory names and file stems.
--verbose                  Enable verbose logging (including skipped non-UTF-8 file notices).
--debug                    Enable debug logging (includes verbose logging).
```

## Patterns format

- One regex pattern per line.
- Empty lines are ignored.
- Lines starting with `#` are ignored.
- Matching is case-insensitive globally.

Example:

```text
# basic literals
test
alpha

# regex patterns
test[0-9]+
usr_[a-z]{4,10}[0-9]{1,4}
```

## Mapping JSON

Each forward run writes one JSON file:

```text
<json-dir>/ai-liberator-map-YYYYMMDDHHMMSS.json
```

It includes:

- Run metadata (`timestamp`, `created_at`, input/pattern paths).
- `processed_files` (input/output path pairs).
- `processed_paths` (input/output pairs including relative paths for path renaming restore).
- `skipped_files` (files not processed, e.g. non-UTF-8 content).
- `token_to_original` dictionary.
- `path_token_to_original` dictionary (tokens created while transforming names).
- `mappings` list (token, original, pattern index, pattern).

## Behavior details

- Token generation is deterministic from `(pattern_index, matched_text)`.
- Different case variants and different concrete matches map to different tokens.
- Replacement never applies overlapping matches.
- Reverse mode replaces tokens by longest token first to avoid substring conflicts.

## Testing

Run the test suite:

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

## Performance notes

The main cost is regex candidate collection (`collect_match_candidates`) on large corpora.

Useful knobs:

- Use `--jobs` for file-level parallelism in forward mode.
- Keep patterns focused to reduce unnecessary scans.
- Prefer directory-level processing when you can parallelize across many files.

## License

GPL-3.0. See [LICENSE](LICENSE).
