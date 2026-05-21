#!/usr/bin/env python3
"""Download the published mouse-brain dataset from Kleshchevnikov et al. 2022.

This is the canonical cell2location demo dataset: 5 Visium sections + paired
single-nucleus RNA-seq reference from `tutorial/mouse_brain_*` on the
Sanger cell2location object store.

Use this dataset to validate your environment / pipeline against published
results before applying cell2location to your own data.

Usage:
    python download_mouse_brain.py --output-dir ./data
    python download_mouse_brain.py --output-dir ./data --components snrna
    python download_mouse_brain.py --output-dir ./data --components visium
"""
import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

BASE = "https://cell2location.cog.sanger.ac.uk/tutorial"

VISIUM_SAMPLES = [
    "ST8059048",  # Visium-28C
    "ST8059049",  # Visium-28D
    "ST8059050",  # Visium-28E
    "ST8059051",  # Visium-29B
    "ST8059052",  # Visium-29C
]

VISIUM_PER_SAMPLE_FILES = [
    "filtered_feature_bc_matrix.h5",
    "filtered_feature_bc_matrix/barcodes.tsv.gz",
    "filtered_feature_bc_matrix/features.tsv.gz",
    "filtered_feature_bc_matrix/matrix.mtx.gz",
    "spatial/tissue_lowres_image.png",
    "spatial/tissue_hires_image.png",
    "spatial/scalefactors_json.json",
    "spatial/tissue_positions_list.csv",
]

SNRNA_FILES = [
    ("mouse_brain_snrna/all_cells_20200625.h5ad", "snrna/all_cells.h5ad"),
    (
        "mouse_brain_snrna/regression_model/"
        "RegressionGeneBackgroundCoverageTorch_65covariates_40532cells_12819genes/sc.h5ad",
        "snrna/sc.h5ad",
    ),
    (
        "mouse_brain_snrna/snRNA_annotation_astro_subtypes_refined59_20200823.csv",
        "snrna/annotation.csv",
    ),
]

INDEX_FILES = [
    ("mouse_brain_visium_data/Visium_mouse.csv", "visium/Visium_mouse.csv"),
]


def download_one(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  exists: {dest.relative_to(dest.parent.parent.parent)} ({dest.stat().st_size:,} bytes)")
        return
    print(f"  downloading: {url}")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        raise RuntimeError(f"failed to download {url}: {e}") from e
    print(f"    wrote {dest.relative_to(dest.parent.parent.parent)} ({dest.stat().st_size:,} bytes)")


def download_visium(output_dir: Path) -> None:
    print(f"Downloading Visium spatial data ({len(VISIUM_SAMPLES)} sections)...")
    for src, dst in INDEX_FILES:
        download_one(f"{BASE}/{src}", output_dir / dst)
    for sample in VISIUM_SAMPLES:
        print(f"  sample: {sample}")
        for f in VISIUM_PER_SAMPLE_FILES:
            url = f"{BASE}/mouse_brain_visium_data/rawdata/{sample}/{f}"
            dest = output_dir / "visium" / "rawdata" / sample / f
            try:
                download_one(url, dest)
            except RuntimeError as e:
                # Some auxiliary spatial files are optional; warn but continue.
                if "spatial/" in f:
                    print(f"    SKIPPED ({e})")
                else:
                    raise


def download_snrna(output_dir: Path) -> None:
    print("Downloading snRNA-seq reference data...")
    for src, dst in SNRNA_FILES:
        download_one(f"{BASE}/{src}", output_dir / dst)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to download into (default: this script's directory).",
    )
    p.add_argument(
        "--components",
        choices=["all", "visium", "snrna"],
        default="all",
        help="Which components to download (default: all).",
    )
    args = p.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    if args.components in ("all", "snrna"):
        download_snrna(out)
    if args.components in ("all", "visium"):
        download_visium(out)

    print("")
    print("Downloaded to:", out)
    print("")
    print("Usage from the spatial-mapping skill:")
    print(f"  spatial_h5ad_path = (build adata from {out / 'visium/rawdata/'} via sc.read_visium)")
    print(f"  ref_h5ad_path     = '{out / 'snrna/all_cells.h5ad'}'")
    print(f"  signatures_csv    = (run step1_reference_signatures.ipynb on the reference)")
    print("")
    print("Reference paper: Kleshchevnikov et al. 2022. doi:10.1038/s41587-021-01139-4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
