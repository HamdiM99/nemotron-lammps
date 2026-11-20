#!/usr/bin/env python3
"""Construit l'interface 3D d'une experience a partir d'un dossier de travail de l'agent.

  python viewer/build_viewer.py runs/p1

Lit trace.jsonl, log.full.lammps et traj.full.lammpstrj, et ecrit dans viewer/ :
  graphene_viewer.html   page autonome, a ouvrir dans un navigateur
  artifact.html          meme contenu sans l'enveloppe html/head/body
Aucune valeur n'est recalculee autrement que par graphene_agent (parse_thermo, linfit).
"""
import argparse
import base64
import json
import re
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import graphene_agent as ga  # noqa: E402


def read_traj(path: Path):
    """Retourne (pas, boites, positions) ; positions[k] = liste plate x,y,z triee par id."""
    steps, boxes, frames = [], [], []
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("ITEM: TIMESTEP"):
            steps.append(int(lines[i + 1]))
            n = int(lines[i + 3])
            (xlo, xhi), (ylo, yhi) = (map(float, lines[i + 5].split()[:2]), map(float, lines[i + 6].split()[:2]))
            boxes.append([round(v, 4) for v in (xlo, xhi, ylo, yhi)])
            cols = lines[i + 8].split()[2:]
            ix, iy, iz, iid = (cols.index(c) for c in ("x", "y", "z", "id"))
            rows = sorted((l.split() for l in lines[i + 9:i + 9 + n]), key=lambda t: int(t[iid]))
            frames.append([float(t[c]) for t in rows for c in (ix, iy, iz)])
            i += 9 + n
        else:
            i += 1
    return steps, boxes, frames


def build(workdir: Path) -> dict:
    trace = [json.loads(l) for l in (workdir / "trace.jsonl").read_text(encoding="utf-8").splitlines() if l]
    tools = [r for r in trace if r["type"] == "tool"]

    def last_ok(name):
        hits = [r["result"] for r in tools if r["name"] == name and r["result"].get("ok")]
        if not hits:
            sys.exit(f"{workdir} : aucun appel {name} reussi dans trace.jsonl")
        return hits[-1]

    structure, parsed = last_ok("build_structure"), last_ok("parse_log")
    spec = dict(last_ok("write_input")["spec"])
    spec["potential_file"] = Path(spec.pop("potential_path")).name     # pas de chemin local dans la page
    if spec["task"] != "tensile":
        sys.exit("l'interface ne couvre que la tache tensile")

    log = (workdir / "log.full.lammps").read_text(errors="replace")
    rows = ga.parse_thermo(log)[-1]
    lx0 = rows[0]["Lx"]
    strain = [(r["Lx"] - lx0) / lx0 for r in rows]
    stress = [-r["Pxx"] * 1e-4 * ga.LZ / ga.THICKNESS for r in rows]       # GPa, comme parse_log
    slope, _ = ga.linfit(strain, stress)
    intercept = sum(stress) / len(stress) - slope * sum(strain) / len(strain)

    steps, boxes, frames = read_traj(workdir / "traj.full.lammpstrj")
    n = len(frames[0]) // 3
    packed = b"".join(struct.pack(f"<{3 * n}h", *(round(v * 100) for v in f)) for f in frames)   # 0,01 A

    def clean(args):
        try:
            return json.loads(args or "{}")
        except json.JSONDecodeError:
            return {"brut": args}

    version = re.search(r"LAMMPS \(([^)]*)\)", log)
    return {
        "meta": {"prompt": trace[0].get("prompt", ""), "model": trace[0].get("model", ""),
                 "lammps": version.group(1) if version else "", "thickness_A": ga.THICKNESS},
        "structure": structure, "spec": spec, "result": parsed["result"], "warnings": parsed["warnings"],
        "steps": [{"name": r["name"], "args": clean(r["args"]), "ok": bool(r["result"].get("ok")),
                   "errors": r["result"].get("errors", [])} for r in tools],
        "curve": {"strain": [round(v, 6) for v in strain], "stress": [round(v, 4) for v in stress],
                  "slope": slope, "intercept": intercept},
        "traj": {"n": n, "count": len(frames), "steps": steps, "boxes": boxes,
                 "pos": base64.b64encode(packed).decode()},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("workdir")
    a = ap.parse_args()
    data = build(Path(a.workdir))
    page = (HERE / "template.html").read_text(encoding="utf-8").replace(
        "/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
    (HERE / "artifact.html").write_text(page, encoding="utf-8")
    (HERE / "graphene_viewer.html").write_text(
        '<!doctype html>\n<html lang="fr"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1"></head><body>\n'
        + page + "\n</body></html>\n", encoding="utf-8")
    # page du depot (GitHub Pages : Settings > Pages > branche main, dossier /docs)
    desc = (f"3D view of a graphene sheet stretched in LAMMPS by an LLM agent: Young's modulus "
            f"{data['result']['young_TPa']} TPa, stress-strain curve and the tool calls the agent made.")
    docs = HERE.parent / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "index.html").write_text(
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<meta name="description" content="{desc}">\n'
        '<meta property="og:type" content="website">\n'
        '<meta property="og:title" content="Graphene Tensile Bench">\n'
        f'<meta property="og:description" content="{desc}">\n'
        '</head><body>\n' + page + "\n</body></html>\n", encoding="utf-8")
    print(f"{data['traj']['count']} images, {data['traj']['n']} atomes -> "
          f"{HERE / 'graphene_viewer.html'} ({len(page) // 1024} ko)")
