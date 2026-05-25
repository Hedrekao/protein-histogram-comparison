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
from Bio.PDB import MMCIFParser, PDBIO, PDBParser, Select
from Bio.PDB.Polypeptide import is_aa


RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.{ext}"


@dataclass
class ProteinData:
    pdb_id: str
    chain_id: str
    residue_count: int
    ca_coords: np.ndarray
    structure_text: str
    format_name: str


def parse_protein_ref(reference: str) -> tuple[str, str | None]:
    if ":" in reference:
        pdb_id, chain_id = reference.split(":", 1)
        chain_id = chain_id.strip() or None
    else:
        pdb_id, chain_id = reference, None
    pdb_id = pdb_id.strip().upper()
    if len(pdb_id) != 4:
        raise ValueError(f"Expected a 4-character PDB id, got {reference!r}")
    return pdb_id, chain_id


def fetch_structure_text(pdb_id: str) -> tuple[str, str]:
    for ext, format_name in (("pdb", "PDB"), ("cif", "mmCIF")):
        url = RCSB_PDB_URL.format(pdb_id=pdb_id.lower(), ext=ext)
        try:
            with urlopen(url) as response:
                return response.read().decode("utf-8"), format_name
        except (HTTPError, URLError):
            continue
    raise RuntimeError(f"Could not download structure for {pdb_id}")


def load_structure(pdb_id: str, text: str, format_name: str):
    handle = io.StringIO(text)
    if format_name == "PDB":
        parser = PDBParser(QUIET=True)
    else:
        parser = MMCIFParser(QUIET=True)
    return parser.get_structure(pdb_id, handle)


def count_ca_atoms(chain) -> int:
    return sum(
        1 for residue in chain if is_aa(residue, standard=False) and "CA" in residue
    )


def choose_chain(structure, preferred_chain_id: str | None = None):
    chains = list(structure.get_chains())
    if not chains:
        raise RuntimeError("No chains found in structure")

    if preferred_chain_id is not None:
        for chain in chains:
            if chain.id == preferred_chain_id:
                if count_ca_atoms(chain) == 0:
                    raise RuntimeError(
                        f"Chain {preferred_chain_id!r} does not contain any C-alpha atoms"
                    )
                return chain
        available = ", ".join(chain.id for chain in chains)
        raise RuntimeError(
            f"Chain {preferred_chain_id!r} not found. Available chains: {available}"
        )

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


def chain_to_pdb_text(chain) -> str:
    class ChainSelect(Select):
        def __init__(self, chain_id: str):
            self.chain_id = chain_id

        def accept_chain(self, chain):
            return chain.id == self.chain_id

        def accept_residue(self, residue):
            return is_aa(residue, standard=False)

    buffer = io.StringIO()
    io_writer = PDBIO()
    io_writer.set_structure(chain.get_parent().get_parent())
    io_writer.save(buffer, select=ChainSelect(chain.id))
    return buffer.getvalue()


def pairwise_ca_distances(coords: np.ndarray) -> np.ndarray:
    if coords.shape[0] < 2:
        raise RuntimeError("At least two residues are required to build a histogram")
    diff = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=-1))
    return distances[np.triu_indices_from(distances, k=1)]


def normalized_histogram(
    distances: np.ndarray, bin_width: float, max_distance: float
) -> tuple[np.ndarray, np.ndarray]:
    upper_edge = np.ceil(max_distance / bin_width) * bin_width
    edges = np.arange(0.0, upper_edge + bin_width, bin_width)
    counts, edges = np.histogram(distances, bins=edges)
    total = counts.sum()
    probabilities = counts / total if total else counts.astype(float)
    return probabilities, edges


def wasserstein_1(hist_a: np.ndarray, hist_b: np.ndarray, bin_width: float) -> float:
    return float(np.sum(np.abs(np.cumsum(hist_a) - np.cumsum(hist_b))) * bin_width)


def histogram_overlap(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    return float(np.sum(np.minimum(hist_a, hist_b)))


def format_distance_summary(
    name_a: str,
    name_b: str,
    bins: np.ndarray,
    hist_a: np.ndarray,
    hist_b: np.ndarray,
    bin_width: float,
) -> str:
    w1 = wasserstein_1(hist_a, hist_b, bin_width)
    overlap = histogram_overlap(hist_a, hist_b)
    return dedent(
        f"""
        <table class="metrics">
          <tr><th>Protein A</th><td>{name_a}</td></tr>
          <tr><th>Protein B</th><td>{name_b}</td></tr>
          <tr><th>Histogram bins</th><td>{len(bins) - 1} bins at {bin_width:.2f} Å</td></tr>
          <tr><th>Wasserstein-1 distance</th><td>{w1:.3f} Å</td></tr>
          <tr><th>Histogram overlap</th><td>{overlap:.3f}</td></tr>
        </table>
        """
    ).strip()


def make_histogram_figure(
    name_a: str,
    name_b: str,
    distances_a: np.ndarray,
    distances_b: np.ndarray,
    bin_width: float,
) -> bytes:
    max_distance = max(float(distances_a.max()), float(distances_b.max()))
    hist_a, edges = normalized_histogram(distances_a, bin_width, max_distance)
    hist_b, _ = normalized_histogram(distances_b, bin_width, max_distance)
    centers = edges[:-1] + bin_width / 2

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
    ax.set_title("C-alpha distance distribution")
    ax.set_xlabel("Distance (Å)")
    ax.set_ylabel("Probability")
    ax.set_xlim(0, edges[-1])
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def compare_histograms(
    distances_a: np.ndarray, distances_b: np.ndarray, bin_width: float
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    max_distance = max(float(distances_a.max()), float(distances_b.max()))
    hist_a, bins = normalized_histogram(distances_a, bin_width, max_distance)
    hist_b, _ = normalized_histogram(distances_b, bin_width, max_distance)
    return (
        wasserstein_1(hist_a, hist_b, bin_width),
        histogram_overlap(hist_a, hist_b),
        hist_a,
        hist_b,
        bins,
    )


def make_viewer_html(pdb_text: str, title: str) -> str:
    view = py3Dmol.view(width=420, height=420)
    view.addModel(pdb_text, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.zoomTo()
    view.setBackgroundColor("white")
    return f"<div class='viewer-title'>{title}</div>{view._make_html()}"


def build_report(
    output_path: Path,
    protein_a: ProteinData,
    protein_b: ProteinData,
    histogram_png: bytes,
    bin_width: float,
) -> None:
    distances_a = pairwise_ca_distances(protein_a.ca_coords)
    distances_b = pairwise_ca_distances(protein_b.ca_coords)
    _, _, hist_a, hist_b, bins = compare_histograms(distances_a, distances_b, bin_width)
    metrics_html = format_distance_summary(
        f"{protein_a.pdb_id}:{protein_a.chain_id}",
        f"{protein_b.pdb_id}:{protein_b.chain_id}",
        bins,
        hist_a,
        hist_b,
        bin_width,
    )
    image_b64 = base64.b64encode(histogram_png).decode("ascii")

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
            .card {{ background: white; border-radius: 16px; padding: 18px; box-shadow: 0 10px 30px rgba(19, 30, 54, 0.08); }}
            .viewer-title {{ font-weight: 700; margin-bottom: 10px; font-size: 1rem; }}
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
              The comparison uses the 1D Wasserstein distance on normalized histograms.
            </p>
            {metrics_html}
          </div>

          <div class="grid two-col" style="margin-bottom: 20px;">
            <div class="card">{make_viewer_html(protein_a.structure_text, f"{protein_a.pdb_id}:{protein_a.chain_id} ({protein_a.format_name}, {protein_a.residue_count} residues)")}</div>
            <div class="card">{make_viewer_html(protein_b.structure_text, f"{protein_b.pdb_id}:{protein_b.chain_id} ({protein_b.format_name}, {protein_b.residue_count} residues)")}</div>
          </div>

          <div class="card">
            <h2>Histogram figure</h2>
            <img class="figure" src="data:image/png;base64,{image_b64}" alt="C-alpha distance histograms" />
          </div>
        </body>
        </html>
        """
    ).strip()

    output_path.write_text(html, encoding="utf-8")


def summarize_protein(
    pdb_ref: str, preferred_chain_id: str | None = None
) -> ProteinData:
    pdb_id, chain_from_ref = parse_protein_ref(pdb_ref)
    selected_chain_id = preferred_chain_id or chain_from_ref
    text, format_name = fetch_structure_text(pdb_id)
    structure = load_structure(pdb_id, text, format_name)
    chain = choose_chain(structure, selected_chain_id)
    coords = extract_ca_coordinates(chain)
    chain_text = chain_to_pdb_text(chain)
    return ProteinData(
        pdb_id=pdb_id,
        chain_id=chain.id,
        residue_count=coords.shape[0],
        ca_coords=coords,
        structure_text=chain_text,
        format_name=format_name,
    )


def build_output_path(output_dir: Path, protein_a: str, protein_b: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{protein_a.replace(':', '_')}_vs_{protein_b.replace(':', '_')}"
    return output_dir / f"{safe_name}.html"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare protein structures using C-alpha distance histograms."
    )
    parser.add_argument(
        "protein_a", help="First protein PDB id, optionally with chain id like 1MBO:A"
    )
    parser.add_argument(
        "protein_b", help="Second protein PDB id, optionally with chain id like 1UBQ:B"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for the generated HTML report",
    )
    parser.add_argument(
        "--bin-width", type=float, default=1.0, help="Histogram bin width in Å"
    )
    args = parser.parse_args()

    protein_a = summarize_protein(args.protein_a)
    protein_b = summarize_protein(args.protein_b)

    distances_a = pairwise_ca_distances(protein_a.ca_coords)
    distances_b = pairwise_ca_distances(protein_b.ca_coords)
    histogram_png = make_histogram_figure(
        f"{protein_a.pdb_id}:{protein_a.chain_id}",
        f"{protein_b.pdb_id}:{protein_b.chain_id}",
        distances_a,
        distances_b,
        args.bin_width,
    )

    output_path = build_output_path(
        Path(args.output_dir), args.protein_a, args.protein_b
    )
    build_report(output_path, protein_a, protein_b, histogram_png, args.bin_width)

    w1, overlap, _, _, _ = compare_histograms(distances_a, distances_b, args.bin_width)
    print(f"Saved report to {output_path}")
    print(f"Wasserstein-1 histogram distance: {w1:.3f} Å")
    print(f"Histogram overlap: {overlap:.3f}")
    print(
        f"Selected chains: {protein_a.pdb_id}:{protein_a.chain_id} and {protein_b.pdb_id}:{protein_b.chain_id}"
    )


if __name__ == "__main__":
    main()
