#!/usr/bin/env python3
"""Test hors score du cas I03 du banc v2 : 0,467 TPa avec R2 = 0,899, bruit ou effet reel ?

I03 (v2) : feuille 5x5 nm, 5 % de lacunes, graine 42, Tersoff, 300 K, dt 0,5 fs, 0,001 /ps,
deformation maximale 2 %. On rejoue ces parametres sans LLM (pipeline deterministe) :
  - graine 42 a 2 % : doit reproduire le resultat du banc ;
  - plusieurs graines a 5 % : on ajuste la pente sur toute la fenetre et sur les 2 premiers %.
Seul le tirage des lacunes change ; la graine des vitesses est fixee dans l'input.

  python bench/i03_seeds.py            # ecrit bench/i03_seeds.csv
"""
import csv
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import graphene_agent as ga  # noqa: E402

SEEDS = [42, 1, 2, 3, 4]
BASE = {"task": "tensile", "potential": "tersoff", "temperature_K": 300, "timestep_fs": 0.5,
        "equil_steps": 10000, "strain_rate_per_ps": 0.001}


def fit(workdir: Path, window: float) -> tuple:
    """Pente contrainte/deformation (TPa) et R2 sur les deformations <= window, comme parse_log."""
    rows = ga.parse_thermo((workdir / "log.full.lammps").read_text(errors="replace"))[-1]
    lx0 = rows[0]["Lx"]
    pts = [((r["Lx"] - lx0) / lx0, -r["Pxx"] * 1e-4 * ga.LZ / ga.THICKNESS) for r in rows]
    pts = [p for p in pts if p[0] <= window + 1e-9]
    slope, r2 = ga.linfit([p[0] for p in pts], [p[1] for p in pts])
    return round(slope / 1000, 3), round(r2, 3), len(pts)


def one(job: tuple) -> dict:
    seed, max_strain = job
    wd = HERE / "runs_i03" / f"seed{seed}_strain{max_strain}"
    out = ga.run_spec({"structure": {"size_x_nm": 5, "size_y_nm": 5, "vacancy_pct": 5, "seed": seed},
                       "input": {**BASE, "max_strain": max_strain}}, str(wd))
    row = {"graine": seed, "max_strain": max_strain, "ok": out.get("ok"),
           "avertissements": " ; ".join(out.get("warnings", []) or out.get("errors", []))}
    if out.get("ok"):
        row["E_parse_log_TPa"], row["R2_parse_log"] = out["result"]["young_TPa"], out["result"]["fit_r2"]
        row["E_fenetre_2pct_TPa"], row["R2_fenetre_2pct"], row["points_2pct"] = fit(wd, 0.02)
    return row


if __name__ == "__main__":
    jobs = [(42, 0.02)] + [(s, 0.05) for s in SEEDS]
    with ThreadPoolExecutor(3) as ex:
        rows = list(ex.map(one, jobs))
    cols = ["graine", "max_strain", "ok", "E_parse_log_TPa", "R2_parse_log", "E_fenetre_2pct_TPa",
            "R2_fenetre_2pct", "points_2pct", "avertissements"]
    with (HERE / "i03_seeds.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"{'graine':>6} {'max':>5} {'E (parse_log)':>14} {'R2':>6} {'E (<= 2 %)':>11} {'R2':>6}  avertissements")
    for r in rows:
        print(f"{r['graine']:>6} {r['max_strain']:>5} {r.get('E_parse_log_TPa', '-'):>14} {r.get('R2_parse_log', '-'):>6} "
              f"{r.get('E_fenetre_2pct_TPa', '-'):>11} {r.get('R2_fenetre_2pct', '-'):>6}  {r['avertissements']}")
    full = [r for r in rows if r["max_strain"] == 0.05 and r["ok"]]
    if len(full) > 1:
        for key, label in (("E_parse_log_TPa", "fenetre 5 %"), ("E_fenetre_2pct_TPa", "fenetre 2 %")):
            v = [r[key] for r in full]
            print(f"{label} sur {len(v)} graines : moyenne {statistics.mean(v):.3f} TPa, ecart-type {statistics.stdev(v):.3f}, "
                  f"min {min(v):.3f}, max {max(v):.3f}")
