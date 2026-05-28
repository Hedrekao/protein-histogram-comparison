from __future__ import annotations

import argparse
import base64
import io
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
import py3Dmol
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa


RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


@dataclass
class ProteinData:
    pdb_id: str
    protein_name: str
    chain_id: str
    residue_count: int
    ca_coords: np.ndarray
    structure_text: str

    def get_display_name(self) -> str:
        return f"{self.pdb_id}:{self.chain_id} ({self.protein_name})"

# Count number of C-alpha atoms in residues of a chain that are recognized as amino acids
def count_ca_atoms(chain) -> int:
    return sum(
        1 for residue in chain if is_aa(residue, standard=False) and "CA" in residue
    )


# If a structure has multiple chains, we select the one with the most C-alpha atoms.
def choose_protein_chain(structure):
    chains = list(structure.get_chains())
    if not chains:
        raise RuntimeError("No chains found in structure")

    ranked = sorted(chains, key=count_ca_atoms, reverse=True)
    selected = ranked[0]
    if count_ca_atoms(selected) == 0:
        raise RuntimeError("No protein chain with C-alpha atoms found")
    return selected


def extract_ca_coordinates(chain) -> np.ndarray:
    coords = []
    for residue in chain:
        if not is_aa(residue, standard=False):
            continue
        if residue.has_id("CA"):
            atom = residue["CA"]
            coords.append(atom.get_coord())
    if not coords:
        raise RuntimeError(f"Chain {chain.id!r} has no C-alpha coordinates")
    return np.asarray(coords, dtype=float)


def pairwise_ca_distances(coords: np.ndarray) -> np.ndarray:
    if coords.shape[0] < 2:
        raise RuntimeError("At least two residues are required to build a histogram")
    # Compute pairwise distances using broadcasting.
    diff = coords[:, None, :] - coords[None, :, :]
    # Euclidean distance between C-alpha atoms
    distances = np.sqrt(np.sum(diff * diff, axis=-1))
    # We only need the upper triangle of the distance matrix, as distances are symmetric and we don't include self-distances.
    return distances[np.triu_indices_from(distances, k=1)]


def create_histogram(distances: np.ndarray, bins: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(distances, bins=bins)
    total = counts.sum()
    probabilities = counts / total
    return probabilities


# Compute the 1D Wasserstein distance between two histograms.
# Wassertain distance is the minimum cost of transforming one distribution into another, where cost is defined as the amount of "mass" moved times the distance it is moved.
def wasserstein_1(hist_a: np.ndarray, hist_b: np.ndarray, bin_width: float) -> float:
    return float(np.sum(np.abs(np.cumsum(hist_a) - np.cumsum(hist_b))) * bin_width)


# Compute histogram overlap as the sum of minimum values in each bin. This gives a measure of how much the two histograms overlap, with 1 meaning complete overlap and 0 meaning no overlap.
def histogram_overlap(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    return float(np.sum(np.minimum(hist_a, hist_b)))


def format_distance_summary(
    name_a: str,
    name_b: str,
    bins: np.ndarray,
    bin_width: float,
    wasserstein_distance: float,
    overlap_ratio: float,
) -> str:
    return dedent(
        f"""
        <table class="metrics">
          <tr><th>Protein A</th><td>{name_a}</td></tr>
          <tr><th>Protein B</th><td>{name_b}</td></tr>
          <tr><th>Histogram bins</th><td>{len(bins) - 1} bins at {bin_width:.2f} Å</td></tr>
          <tr><th>Wasserstein-1 distance</th><td>{wasserstein_distance:.3f} Å</td></tr>
          <tr><th>Histogram overlap</th><td>{overlap_ratio:.3f}</td></tr>
        </table>
        """
    ).strip()


def figure_to_png(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def make_histogram_figure(
    name: str,
    hist: np.ndarray,
    bins: np.ndarray,
    bin_width: float,
    color: str,
) -> bytes:
    centers = bins[:-1] + bin_width / 2
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    ax.bar(centers, hist, width=bin_width * 0.92, color=color, alpha=0.9)
    ax.set_title(f"{name} distance distribution")
    ax.set_xlabel("Distance [Å]")
    ax.set_ylabel("Probability")
    ax.set_xlim(0, bins[-1])
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    return figure_to_png(fig)


def make_histogram_overlay_figure(
    name_a: str,
    name_b: str,
    hist_a: np.ndarray,
    hist_b: np.ndarray,
    bins: np.ndarray,
    bin_width: float,
) -> bytes:
    centers = bins[:-1] + bin_width / 2

    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    ax.bar(
        centers,
        hist_a,
        width=bin_width * 0.92,
        alpha=0.68,
        label=name_a,
        color="#2f6df6",
    )
    ax.bar(
        centers,
        hist_b,
        width=bin_width * 0.92,
        alpha=0.55,
        label=name_b,
        color="#f45b69",
    )
    ax.set_title("C-alpha distance distributions")
    ax.set_xlabel("Distance [Å]")
    ax.set_ylabel("Probability")
    ax.set_xlim(0, bins[-1])
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    return figure_to_png(fig)


def make_viewer_html(pdb_text: str, title: str) -> str:
    view = py3Dmol.view(width=420, height=420)
    view.addModel(pdb_text, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.zoomTo()
    view.setBackgroundColor("white")
    return f"<div class='viewer-title'>{title}</div>{view._make_html()}"


def build_html_report(
    output_path: Path,
    protein_a: ProteinData,
    protein_b: ProteinData,
    bins: np.ndarray,
    bin_width: float,
    wasserstein_distance: float,
    overlap_ratio: float,
    histogram_a_png: bytes,
    histogram_b_png: bytes,
    overlay_png: bytes,
) -> None:
    name_a = protein_a.get_display_name()
    name_b = protein_b.get_display_name()
    metrics_html = format_distance_summary(
        name_a,
        name_b,
        bins,
        bin_width,
        wasserstein_distance,
        overlap_ratio,
    )
    histogram_a_b64 = base64.b64encode(histogram_a_png).decode("ascii")
    histogram_b_b64 = base64.b64encode(histogram_b_png).decode("ascii")
    overlay_b64 = base64.b64encode(overlay_png).decode("ascii")

    html = dedent(
        f"""
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <title>Protein histogram comparison</title>
          <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 24px; background: #f7f8fc; color: #172033; }}
            h1, h2, h3 {{ margin-top: 0; }}
            .grid {{ display: grid; gap: 20px; }}
            .two-col {{ grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); }}
            .card {{ background: white; border-radius: 16px; padding: 18px; box-shadow: 0 10px 30px rgba(19, 30, 54, 0.08); display: flex; flex-direction: column; align-items: center }}
            .viewer-title {{ font-weight: 700; margin-bottom: 10px; font-size: 1rem; display: flex; justify-content: center }}
            .metrics {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
            .metrics th, .metrics td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #e6eaf1; vertical-align: top; }}
            .metrics th {{ width: 200px; color: #4f5b76; font-weight: 600; }}
            .figure {{ width: 100%; max-width: 1000px; display: block; margin: 0 auto; border-radius: 12px; }}
            .note {{ color: #4f5b76; line-height: 1.5; }}
          </style>
        </head>
        <body>
          <div class="card" style="margin-bottom: 20px;">
            <h1>Protein C-alpha distance comparison</h1>
            <p class="note">
              Histograms are built from all pairwise distances between C-alpha atoms in the selected chain.
              The report shows individual histograms first, then the overlay comparison.
            </p>
            {metrics_html}
          </div>

          <div class="grid two-col" style="margin-bottom: 20px;">
            <div class="card">{make_viewer_html(protein_a.structure_text, f"{name_a} - ({protein_a.residue_count} residues)")}</div>
            <div class="card">{make_viewer_html(protein_b.structure_text, f"{name_b} - ({protein_b.residue_count} residues)")}</div>
          </div>

          <div class="card" style="margin-bottom: 20px;">
            <h2>Histogram - {name_a}</h2>
            <img class="figure" src="data:image/png;base64,{histogram_a_b64}" alt="Histogram for {name_a}" />
          </div>

          <div class="card" style="margin-bottom: 20px;">
            <h2>Histogram - {name_b}</h2>
            <img class="figure" src="data:image/png;base64,{histogram_b_b64}" alt="Histogram for {name_b}" />
          </div>

          <div class="card">
            <h2>Overlay comparison</h2>
            <img class="figure" src="data:image/png;base64,{overlay_b64}" alt="Overlay of C-alpha distance histograms" />
          </div>
        </body>
        </html>
        """
    ).strip()

    output_path.write_text(html, encoding="utf-8")


def get_protein_data(pdb_id: str) -> ProteinData:
    pdb_id = pdb_id.strip().upper()
    if len(pdb_id) != 4:
        raise ValueError(f"Expected a 4-character PDB id, got {pdb_id!r}")
    db_url = RCSB_PDB_URL.format(pdb_id=pdb_id.lower())
    try:
        with urlopen(db_url) as response:
            text_data = response.read().decode("utf-8")
    except (HTTPError, URLError):
        raise RuntimeError(f"Could not download structure for {pdb_id}")

    handle = io.StringIO(text_data)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, handle)
    chain = choose_protein_chain(structure)
    coords = extract_ca_coordinates(chain)

    protein_name = structure.header.get("name", "Unknown")
    # If the structure has a compound description with a molecule name, use that as the protein name instead, as it is often more short and descriptive than the header name.
    if 'compound' in structure.header and '1' in structure.header['compound'] and 'molecule' in structure.header['compound']['1']:
        protein_name = structure.header['compound']['1']['molecule']

    return ProteinData(
        pdb_id=pdb_id,
        protein_name=protein_name,
        chain_id=chain.id,
        residue_count=coords.shape[0],
        ca_coords=coords,
        structure_text=text_data,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare protein structures using C-alpha distance histograms."
    )
    parser.add_argument("protein_a", help="First protein PDB id")
    parser.add_argument("protein_b", help="Second protein PDB id")
    parser.add_argument(
        "--bin-width", type=float, default=1.0, help="Histogram bin width in Å"
    )
    args = parser.parse_args()

    # get protein data by downloading PDB files and extracting C-alpha coordinates
    protein_a = get_protein_data(args.protein_a)
    protein_b = get_protein_data(args.protein_b)

    # compute pairwise C-alpha distances and histograms for both proteins
    distances_a = pairwise_ca_distances(protein_a.ca_coords)
    distances_b = pairwise_ca_distances(protein_b.ca_coords)

    # Get bins so that both histograms use the same bin edges, and the last bin includes the maximum distance observed in either protein.
    max_distance = max(float(distances_a.max()), float(distances_b.max()))
    max_bin = np.ceil(max_distance / args.bin_width) * args.bin_width
    bins = np.arange(0.0, max_bin + args.bin_width, args.bin_width)

    # create histograms data
    hist_a = create_histogram(distances_a, bins)
    hist_b = create_histogram(distances_b, bins)

    # compute histogram comparison metrics
    wasserstein_distance = wasserstein_1(hist_a, hist_b, args.bin_width)
    overlap_ratio = histogram_overlap(hist_a, hist_b)

    protein_a_display_name = protein_a.get_display_name()
    protein_b_display_name = protein_b.get_display_name()

    histogram_a_png = make_histogram_figure(
        protein_a_display_name, hist_a, bins, args.bin_width, "#2f6df6"
    )
    histogram_b_png = make_histogram_figure(
        protein_b_display_name, hist_b, bins, args.bin_width, "#f45b69"
    )
    overlay_png = make_histogram_overlay_figure(
        protein_a_display_name,
        protein_b_display_name,
        hist_a,
        hist_b,
        bins,
        args.bin_width,
    )

    # make sure outputs dir exists
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    output_filename = outputs_dir / f"{protein_a.pdb_id}_vs_{protein_b.pdb_id}.html"

    build_html_report(output_filename, protein_a, protein_b, bins, args.bin_width, wasserstein_distance, overlap_ratio, histogram_a_png, histogram_b_png, overlay_png)

    print(f"Saved HTML report to {output_filename}")
    print(f"Wasserstein-1 histogram distance: {wasserstein_distance:.3f} Å")
    print(f"Histogram overlap: {overlap_ratio:.3f}")
    print(
        f"Selected chains: {protein_a.pdb_id}:{protein_a.chain_id} and {protein_b.pdb_id}:{protein_b.chain_id}"
    )


if __name__ == "__main__":
    main()
