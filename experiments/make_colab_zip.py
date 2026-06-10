"""Build a minimal zip for the Colab neural run: code + the two parquets per
dataset that ``extract_features`` / ``load_bundle`` need. Excludes data/raw,
jsonl backups, sbert cache, semantic_interaction, .git — keeps it ~1.5 MB."""
from __future__ import annotations

import glob
import os
import zipfile

ROOT = os.getcwd()
OUT = os.path.join(os.path.dirname(ROOT), "Explainable-Hybrid-ASAG-colab.zip")
PREFIX = "Explainable-Hybrid-ASAG"

PATTERNS = [
    "pyproject.toml", "Makefile", "README.md", "CLAUDE.md",
    "configs/**/*.yaml", "configs/**/*.yml",
    "src/**/*.py",
    "notebooks/02_neural_colab.ipynb",
]


def main() -> None:
    files: set[str] = set()
    for p in PATTERNS:
        files.update(glob.glob(p, recursive=True))
    for d in glob.glob("data/processed/*/"):
        for f in ("encoder.parquet", "features.parquet"):
            fp = os.path.join(d, f)
            if os.path.exists(fp):
                files.add(fp)
    files = sorted(f for f in files if os.path.isfile(f))

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            arc = (PREFIX + "/" + f).replace("\\", "/")
            z.write(f, arcname=arc)

    size = os.path.getsize(OUT) / 1e6
    ds = sorted({f.replace("\\", "/").split("/")[2]
                 for f in files if f.replace("\\", "/").startswith("data/processed")})
    print(f"wrote {OUT}")
    print(f"size  {size:.2f} MB, {len(files)} files")
    print("datasets:", ds)


if __name__ == "__main__":
    main()
