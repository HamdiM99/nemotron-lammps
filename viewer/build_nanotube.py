#!/usr/bin/env python3
"""Construit l'application des navettes de nanotubes a partir d'un dossier de nanotube_lab.

  python viewer/build_nanotube.py runs_nt

Lit results.json (proprietes statiques) et ecrit :
  docs/nanotube.html                page autonome, publiee avec le depot
  viewer/nanotube_artifact.html     meme contenu sans l'enveloppe html/head/body
"""
import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import nanotube_lab as nt  # noqa: E402


def build(workdir: Path) -> dict:
    res = json.loads((workdir / "results.json").read_text())
    # la page ne publie que les proprietes statiques : les essais de fermeture et de rotation n'y figurent pas
    keep = {k: v for k, v in res.items() if k in ("static", "charges", "force", "oscillator")}
    keep["warnings"] = nt.plausibility(keep)
    data = {"results": keep, "bounds": nt.BOUNDS}
    st, gaps = keep.get("static", {}), keep.get("force", {}).get("gaps", [])
    if st and gaps:
        # geometrie de depart des deux navettes, charges calculees a la tension de seuil du plus petit ecart
        n, cells, gap, volt = st["n_inner"], st["cells"], gaps[0]["gap_A"], gaps[0].get("threshold_V") or 10.0
        groups = nt.head_to_head(n, cells, gap)
        pos = np.vstack(groups)
        sizes = np.cumsum([0] + [len(g) for g in groups])
        q = nt.solve_charges(pos, np.where(np.arange(len(pos)) < sizes[2], volt / 2, -volt / 2))
        inner, outer = np.vstack([groups[0], groups[2]]), np.vstack([groups[1], groups[3]])
        pack = lambda arr: base64.b64encode(np.round(arr * 100).astype("<i2").tobytes()).decode()  # noqa: E731
        data["geom"] = {"gap_A": gap, "voltage_V": volt, "n_inner": len(inner), "n_outer": len(outer),
                        "inner": pack(inner), "outer": pack(outer),
                        "q_inner": [round(float(v), 5) for v in np.r_[q[sizes[0]:sizes[1]], q[sizes[2]:sizes[3]]]]}
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
    print(f"{docs / 'nanotube.html'} ({len(page) // 1024} ko), vue 3D : {'oui' if 'geom' in data else 'non'}")
