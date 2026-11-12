# Benchmarks and cross-checks

Results: [docs/benchmarks.md](../docs/benchmarks.md).

* `speed_llg.py`, `speed_mc.py` — magex timings (1 and 2 threads).
* `crosscheck_vampire_mc.py` — export `sc.json` with `magex export sc.json --to vampire --outdir vamp`, run
  VAMPIRE in `vamp/` (curie-temperature program), then this script compares m(T) with magex.
* `crosscheck_uppasd_mc.sh` — run from an UppASD export directory (`magex export sc.json --to uppasd --outdir upp`),
  with `UPPASD=/path/to/uppasd`.
* `crosscheck_spirit_energy.py` — needs Spirit's Python package (`SPIRIT_PYTHON=/path/to/spirit/core/python`).
