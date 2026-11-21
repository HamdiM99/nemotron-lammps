#!/usr/bin/env python3
"""Construit l'application des navettes de nanotubes a partir d'un dossier de nanotube_lab.

  python viewer/build_nanotube.py runs_nt

Lit results.json, close.data (charges) et close.lammpstrj (trajectoire), et ecrit :
  docs/nanotube.html                page autonome, publiee avec le depot
  viewer/nanotube_artifact.html     meme contenu sans l'enveloppe html/head/body
"""
import argparse
import base64
import json
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import nanotube_lab as nt  # noqa: E402


def read_charges(path: Path) -> list:
    """Charges par atome, dans l'ordre des identifiants, lues dans la section Atoms # charge."""
    lines = path.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("Atoms"))
    rows = [l.split() for l in lines[start + 2:] if l.strip()]
    rows.sort(key=lambda t: int(t[0]))
    return [float(t[2]) for t in rows]


def build(workdir: Path) -> dict:
    res = json.loads((workdir / "results.json").read_text())
    data = {"results": res, "bounds": nt.BOUNDS}
    if (workdir / "close.lammpstrj").exists() and res.get("close", {}).get("ok"):
        steps, frames, types = nt.read_dump(workdir / "close.lammpstrj")
        q = read_charges(workdir / "close.data")
        inner = [i for i, t in enumerate(types) if t in (1, 3)]
        outer = [i for i, t in enumerate(types) if t in (2, 4)]
        pack = lambda arr: base64.b64encode(struct.pack(f"<{arr.size}h", *(round(v * 100) for v in arr.ravel()))).decode()  # noqa: E731
        data["traj"] = {
            "n_inner": len(inner), "n_outer": len(outer), "frames": len(frames),
            "inner": pack(frames[:, inner, :]),               # coeurs : toutes les images, 0,01 A
            "outer": pack(frames[0][outer, :]),               # parois figees : une seule image
            "q_inner": [round(q[i], 5) for i in inner],
            "side_inner": [0 if types[i] == 1 else 1 for i in inner],
        }
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("workdir")
    a = ap.parse_args()
    data = build(Path(a.workdir))
    page = (HERE / "nanotube_template.html").read_text(encoding="utf-8").replace(
        "/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
    (HERE / "nanotube_artifact.html").write_text(page, encoding="utf-8")
    desc = ("Electromechanical properties of two head-to-head carbon nanotube shuttles, computed with LAMMPS "
            "at reduced size: charge distribution, pull-in voltage, interlayer barriers, telescoping oscillation.")
    docs = HERE.parent / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "nanotube.html").write_text(
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<meta name="description" content="{desc}">\n<meta property="og:type" content="website">\n'
        '<meta property="og:title" content="Nanotube Shuttle Lab">\n'
        f'<meta property="og:description" content="{desc}">\n</head><body>\n' + page + "\n</body></html>\n",
        encoding="utf-8")
    print(f"{docs / 'nanotube.html'} ({len(page) // 1024} ko), trajectoire : {'oui' if 'traj' in data else 'non'}")
