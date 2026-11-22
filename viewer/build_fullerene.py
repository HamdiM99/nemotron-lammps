#!/usr/bin/env python3
"""Construit la page de la navette a fullerene a partir d'un dossier de fullerene_shuttle.

  python viewer/build_fullerene.py runs_fs [--run fly_capped_flex_16V]

Lit results.json et <run>.lammpstrj (positions et charges par image), et ecrit :
  docs/fullerene.html               page autonome, publiee avec le depot
  viewer/fullerene_artifact.html    meme contenu sans l'enveloppe html/head/body
"""
import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import fullerene_shuttle as fs  # noqa: E402
import nanotube_lab as nt  # noqa: E402

Q_SCALE = 20000          # charges stockees en entiers 16 bits : 1 unite = 5e-5 e


def read_dump(path: Path) -> tuple:
    """(pas, positions[image][atome][xyz], charges[image][atome], types) ; atomes tries par id."""
    steps, xyz, q, types = [], [], [], None
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("ITEM: TIMESTEP"):
            n = int(lines[i + 3])
            if i + 9 + n > len(lines):
                break                                            # derniere image incomplete
            cols = lines[i + 8].split()[2:]
            arr = np.array([l.split() for l in lines[i + 9:i + 9 + n]], dtype=float)
            arr = arr[np.argsort(arr[:, cols.index("id")])]
            steps.append(int(lines[i + 1]))
            types = arr[:, cols.index("type")].astype(int)
            xyz.append(arr[:, [cols.index(c) for c in "xyz"]])
            q.append(arr[:, cols.index("q")])
            i += 9 + n
        else:
            i += 1
    return steps, np.array(xyz), np.array(q), types


def crossings(run: dict) -> list:
    """Une entree par vol libre : instants de depart et d'arrivee, charge emportee, sens."""
    t, z, q, link = run["time_ps"], run["z_A"], run["q_ball_e"], run["link"]
    out, k = [], 0
    while k < len(link):
        if link[k] == "-":
            j = k
            while j + 1 < len(link) and link[j + 1] == "-":
                j += 1
            if j + 1 < len(link):                                # vol termine par une arrivee
                out.append({"release_ps": t[k], "arrival_ps": t[j + 1], "q_e": q[k],
                            "to": "left" if z[j] < z[k] else "right"})
            k = j + 1
        else:
            k += 1
    return out


def long_run(workdir: Path, res: dict, key: str) -> dict | None:
    """Suite du meme vol : traversees et ecart du C60 a l'axe, image par image."""
    run = res.get(key)
    if not run or not (workdir / f"{key}.lammpstrj").exists():
        return None
    steps, xyz, _, types = read_dump(workdir / f"{key}.lammpstrj")
    ball = xyz[:, types == 3, :].mean(axis=1)
    return {"simulated_ps": run["simulated_ps"], "voltage_V": run["voltage_V"], "crossings": crossings(run),
            "t_ps": [round(s * nt.DT_PS, 2) for s in steps],
            "off_axis_A": np.hypot(ball[:, 0], ball[:, 1]).round(2).tolist()}


def agent_runs(dirs: list) -> list:
    """Vols demandes par le LLM, lus dans trace.jsonl : parametres choisis et verdict ecrit par le code."""
    out = []
    for label, judge, d in dirs:
        trace = Path(d) / "trace.jsonl"
        if not trace.exists():
            continue
        flights, lowest = [], None
        for line in trace.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec["type"] != "tool":
                continue
            if rec["name"] == "report":
                lowest = rec["result"].get("lowest_voltage_crossed_intact_V")
            if rec["name"] != "fly":
                continue
            a, r = json.loads(rec["args"]), rec["result"]
            flights.append({"voltage_V": a.get("voltage_V"), "capped": a.get("capped", True), "flexible": a.get("flexible", False),
                            "ps": round(a.get("steps", 8000) / 1000, 1),
                            "verdict": r.get("verdict") if r.get("ok") else "refused by the code: threshold not computed"})
        out.append({"label": label, "judge": judge, "flights": flights, "lowest_V": lowest})
    return out


def build(workdir: Path, key: str | None) -> dict:
    res = json.loads((workdir / "results.json").read_text())
    # par defaut : le plus long des vols flexibles a calottes
    key = key or max((k for k in res if k.startswith("fly_capped_flex")), key=lambda k: res[k].get("simulated_ps", 0))
    run = res[key]
    steps, xyz, q, types = read_dump(workdir / f"{key}.lammpstrj")
    pack = lambda arr, s: base64.b64encode(np.round(arr * s).astype("<i2").tobytes()).decode()  # noqa: E731
    cross = crossings(run)
    thr = res.get("threshold_capped" if run.get("capped") else "threshold", {})
    period = None
    same = [c["release_ps"] for c in cross if c["to"] == "left"]
    if len(same) >= 2:
        period = (same[-1] - same[0]) / (len(same) - 1)
    return {"key": key, "run": run, "well": res.get("well", {}), "threshold": {k: v for k, v in thr.items() if k != "scan"},
            "crossings": cross, "period_ps": period, "warnings": res.get("warnings", []),
            "bounds": fs.BOUNDS, "contact_A": fs.CONTACT_A,
            "long": long_run(workdir, res, key + "_200ps"),
            "agent": agent_runs([("Run 1", "first judge", "runs_fs_agent"), ("Run 2", "hardened judge", "runs_fs_agent_v2")]),
            "traj": {"frames": len(xyz), "atoms": int(xyz.shape[1]), "n_ball": int((types == 3).sum()),
                     "n_left": int((types == 1).sum()), "frame_ps": [round(s * nt.DT_PS, 3) for s in steps],
                     "xyz": pack(xyz, 100), "q": pack(q, Q_SCALE), "q_scale": Q_SCALE}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("workdir")
    ap.add_argument("--run", default=None, help="cle du vol dans results.json (defaut : premier vol flexible a calottes)")
    a = ap.parse_args()
    data = build(Path(a.workdir), a.run)
    page = (HERE / "fullerene_template.html").read_text(encoding="utf-8").replace(
        "/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
    (HERE / "fullerene_artifact.html").write_text(page, encoding="utf-8")
    desc = ("A C60 fullerene carries charge back and forth between two capped carbon nanotubes held at opposite "
            "voltages. LAMMPS molecular dynamics with charges recomputed in flight, reduced-size model.")
    docs = HERE.parent / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "fullerene.html").write_text(
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<meta name="description" content="{desc}">\n<meta property="og:type" content="website">\n'
        '<meta property="og:title" content="Fullerene Charge Shuttle">\n'
        f'<meta property="og:description" content="{desc}">\n</head><body>\n' + page + "\n</body></html>\n",
        encoding="utf-8")
    print(f"{docs / 'fullerene.html'} ({len(page) // 1024} ko), {data['traj']['frames']} images, "
          f"{len(data['crossings'])} traversees")
