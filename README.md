# Protein Histogram Comparison

This project downloads protein structures from the RCSB PDB, extracts C-alpha atoms, builds a pairwise residue-distance histogram, and compares two structures with the 1D Wasserstein distance.

## What it produces

- A 3D visualization for each selected protein chain
- A histogram figure of the C-alpha distance distributions
- A small HTML report with the comparison metrics

## Usage

Run it with two PDB ids, optionally with chain ids:

```bash
uv run python main.py 1UBQ 1MBO
uv run python main.py 1UBQ:A 1MBO:A
```

The report is written to `outputs/<protein_a>_vs_<protein_b>.html`.

## Method

For each protein chain:

1. Keep only amino-acid residues.
2. Use the C-alpha atom as the residue representative.
3. Compute all pairwise distances between C-alpha atoms.
4. Convert those distances into a normalized histogram.

For comparison, the app uses:

- `Wasserstein-1 distance` between the two normalized histograms
- `Histogram overlap` as a simple similarity score

## How to interpret it

This method is useful as a quick coarse comparison, but it is not enough by itself to prove structural equivalence.

Good example pairs for discussion:

- `1UBQ` vs `1UBQ`: identical structure, so the Wasserstein distance is `0.000 Å` and the overlap is `1.000`.
- `1UBQ` vs `1MBO` (myoglobin): the report I ran gave a Wasserstein distance of `4.786 Å` with overlap `0.703`, showing partial similarity even though the folds are different.
- `1UBQ` vs a very elongated or multi-domain protein: the histogram will usually diverge more clearly and is a better case for separating global shape differences.

The main limitation is that many distinct folds can share similar distance distributions, so the histogram comparison is best treated as a fast screening feature rather than a definitive structural similarity metric. It works best as a coarse shape descriptor, not as a replacement for alignment-based structural comparison.
