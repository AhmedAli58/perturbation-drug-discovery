"""
Download STRING human PPI network and filter to genes relevant for the
perturbation model (HVGs + perturbation target genes).

Source  : STRING v12.0 — https://string-db.org
Species : Homo sapiens (taxon 9606)
Output  : data/external/string_ppi_edges.tsv   (geneA  geneB  weight)

Strategy
--------
1. Collect all gene symbols we care about:
   - 2000 HVG genes from the processed AnnData
   - all unique target genes parsed from perturbation names

2. POST to STRING "network" API → returns interactions between them.
   Chunked into batches of 500 because of URL-length limits.

3. Filter: combined_score >= 700 (high confidence).

4. Save TSV for the GCN model to consume.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
import urllib.parse
import urllib.request
from io import StringIO
from pathlib import Path

import scanpy as sc

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "norman2019_processed.h5ad"
OUT_PATH       = PROJECT_ROOT / "data" / "external" / "string_ppi_edges.tsv"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

MIN_SCORE  = 700   # STRING high-confidence threshold
CHUNK_SIZE = 500   # max genes per API call
SPECIES    = "9606"
STRING_API = "https://string-db.org/api/tsv/network"

# ---------------------------------------------------------------------------
# 1. Collect relevant genes
# ---------------------------------------------------------------------------
logger.info("Loading AnnData to extract gene lists …")
adata = sc.read_h5ad(PROCESSED_PATH)

hvg_genes: list[str] = list(adata.var_names)

pert_genes: set[str] = set()
for p in adata.obs["perturbation"].astype(str).unique():
    if p == "control":
        continue
    for g in p.split("_"):
        pert_genes.add(g)

all_genes = sorted(set(hvg_genes) | pert_genes)
logger.info(
    "HVGs: %d | pert target genes: %d | union: %d",
    len(hvg_genes), len(pert_genes), len(all_genes),
)

# ---------------------------------------------------------------------------
# 2. Query STRING API in chunks
# ---------------------------------------------------------------------------
def _query_string(genes: list[str]) -> list[tuple[str, str, int]]:
    """Query STRING network API for a list of gene symbols.
    Returns list of (geneA, geneB, combined_score).
    """
    params = {
        "identifiers":     "%0d".join(genes),   # \r separator
        "species":         SPECIES,
        "required_score":  str(MIN_SCORE),
        "caller_identity": "perturbation_drug_discovery",
    }
    data = urllib.parse.urlencode(params).encode()
    req  = urllib.request.Request(STRING_API, data=data)

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                content = resp.read().decode()
            break
        except Exception as exc:
            logger.warning("Attempt %d failed: %s", attempt + 1, exc)
            time.sleep(5 * (attempt + 1))
    else:
        logger.error("All retries exhausted for chunk; skipping.")
        return []

    edges: list[tuple[str, str, int]] = []
    reader = csv.DictReader(StringIO(content), delimiter="\t")
    for row in reader:
        # STRING returns preferredName_A / preferredName_B
        a = row.get("preferredName_A", "").strip()
        b = row.get("preferredName_B", "").strip()
        try:
            score = int(row.get("score", "0").strip().replace(".", ""))
        except ValueError:
            # score may be a float string like "0.900"
            try:
                score = int(float(row.get("score", "0")) * 1000)
            except ValueError:
                continue
        if a and b and score >= MIN_SCORE:
            edges.append((a, b, score))
    return edges


all_edges: dict[frozenset, int] = {}   # deduplicate (A,B) == (B,A)

chunks = [all_genes[i:i + CHUNK_SIZE] for i in range(0, len(all_genes), CHUNK_SIZE)]
logger.info("Querying STRING in %d chunk(s) of ≤%d genes …", len(chunks), CHUNK_SIZE)

for i, chunk in enumerate(chunks, 1):
    logger.info("  Chunk %d/%d  (%d genes) …", i, len(chunks), len(chunk))
    edges = _query_string(chunk)
    for a, b, w in edges:
        key = frozenset({a, b})
        all_edges[key] = max(all_edges.get(key, 0), w)
    logger.info("    → %d edges so far (cumulative unique)", len(all_edges))
    time.sleep(1)   # be polite to the STRING servers

# ---------------------------------------------------------------------------
# 3. Save TSV
# ---------------------------------------------------------------------------
hvg_set  = set(hvg_genes)
n_hvg_hvg = 0
n_total   = 0

with open(OUT_PATH, "w") as f:
    f.write("geneA\tgeneB\tweight\n")
    for key, w in sorted(all_edges.items(), key=lambda x: -x[1]):
        a, b = tuple(key)
        # Normalise score to [0, 1]
        weight = round(w / 1000.0, 4)
        f.write(f"{a}\t{b}\t{weight}\n")
        n_total += 1
        if a in hvg_set and b in hvg_set:
            n_hvg_hvg += 1

logger.info("Saved %d total edges → %s", n_total, OUT_PATH)
logger.info("  HVG–HVG edges (used for GCN)  : %d", n_hvg_hvg)
logger.info("  Edges involving pert genes     : %d", n_total - n_hvg_hvg)

# Save a small stats JSON alongside
stats = {
    "total_edges":          n_total,
    "hvg_hvg_edges":        n_hvg_hvg,
    "min_score":            MIN_SCORE,
    "n_hvg_genes":          len(hvg_genes),
    "n_pert_target_genes":  len(pert_genes),
    "n_genes_queried":      len(all_genes),
}
(OUT_PATH.parent / "string_ppi_stats.json").write_text(json.dumps(stats, indent=2))
logger.info("STRING PPI network download complete.")
