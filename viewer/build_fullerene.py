#!/usr/bin/env python3
"""Construit la page de la navette a fullerene a partir d'un dossier de fullerene_shuttle.

  python viewer/build_fullerene.py runs_fs --run fly_capped_flex_16V_45ps

Lit results.json et <run>.lammpstrj (positions et charges par image), plus les traces et trajectoires des
sessions de l'agent (runs_fs_agent*/), et ecrit :
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
AGENT_FRAMES = 70        # images gardees par vol de l'agent, tubes figes (seul le C60 est stocke image par image)
AGENT_FRAMES_FLEX = 36   # tubes flexibles : tous les atomes sont stockes a chaque image, on en garde moins
AGENT_DIRS = [("Run 1", "first judge", "runs_fs_agent"), ("Run 2", "hardened judge", "runs_fs_agent_v2"),
              ("Run 3", "hardened judge, 10 flights, asked to make it oscillate", "runs_fs_agent_v3"),
              ("Run 4", "same, asked for three crossings in one long flight", "runs_fs_agent_v4")]


def pack(arr: np.ndarray, scale: float) -> str:
    return base64.b64encode(np.round(arr * scale).astype("<i2").tobytes()).decode()


def read_dump(path: Path) -> tuple:
    """(pas, positions[image][atome][xyz], charges[image][atome], types) ; atomes tries par id."""
    steps, xyz, q, types = [], [], [], None
    lines = path.read_text().splitlines()
    i, n0 = 0, None
    while i < len(lines):
        if lines[i].startswith("ITEM: TIMESTEP"):
            n = int(lines[i + 3])
            n0 = n0 or n
            if i + 9 + n > len(lines) or n != n0:
                break                                            # image incomplete, ou atomes perdus
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


def period(cross: list) -> float | None:
    same = [c["release_ps"] for c in cross if c["to"] == "left" and c.get("on_axis", True)]
    return (same[-1] - same[0]) / (len(same) - 1) if len(same) >= 2 else None


def traj(steps: list, xyz: np.ndarray, q: np.ndarray, types: np.ndarray, static: bool, max_frames: int | None) -> dict:
    """Trajectoire embarquee. static : tubes figes, on ne stocke leurs positions qu'une fois."""
    idx = np.arange(len(steps)) if not max_frames or len(steps) <= max_frames else \
        np.unique(np.linspace(0, len(steps) - 1, max_frames).round().astype(int))
    ball = types == 3
    out = {"frames": len(idx), "atoms": int(xyz.shape[1]), "n_ball": int(ball.sum()), "static": static,
           "frame_ps": [round(steps[k] * nt.DT_PS, 3) for k in idx], "q": pack(q[idx], Q_SCALE), "q_scale": Q_SCALE}
    if static:
        out["xyz"], out["tube"] = pack(xyz[idx][:, ball, :], 100), pack(xyz[0][~ball], 100)
    else:
        out["xyz"] = pack(xyz[idx], 100)
    return out


def run_from_dump(steps: list, xyz: np.ndarray, q: np.ndarray, types: np.ndarray, args: dict) -> dict:
    """Series temporelles d'un vol de l'agent, recalculees a partir de sa trajectoire (la trace n'en garde qu'un resume)."""
    ball, left, right = types == 3, types == 1, types == 2
    t = np.array(steps) * nt.DT_PS
    z = xyz[:, ball, 2].mean(axis=1)
    qb = q[:, ball].sum(axis=1)
    link = [fs.link_of(xyz[k][ball], xyz[k][left], xyz[k][right]) or "-" for k in range(len(steps))]
    speed = np.abs(np.gradient(z, t)) * 100.0 if len(t) > 2 else np.zeros(len(t))
    free = [k for k, l in enumerate(link) if l == "-"]
    run = {"time_ps": t.round(3).tolist(), "z_A": z.round(3).tolist(), "q_ball_e": qb.round(5).tolist(), "link": link,
           "simulated_ps": round(float(t[-1]), 3), "steps": int(args.get("steps", 8000)), "voltage_V": args["voltage_V"],
           "gap_A": 20.0, "capped": args.get("capped", True), "flexible": args.get("flexible", False), "update": 25,
           "max_speed_m_s": round(float(speed.max()), 1),
           "max_kinetic_eV": round(0.5 * fs.MASS_C60_KG * float(speed.max()) ** 2 / 1.602177e-19, 2),
           "flight_ps": None, "charge_drift_e": None}
    if free:
        end = next((k for k in range(free[0], len(link)) if link[k] != "-"), len(link))
        run["flight_ps"] = round(float(t[end - 1] - t[free[0]]), 3)
        run["charge_drift_e"] = float(np.abs(qb[free[0]:end] - qb[free[0]]).max())
    return run


def agent_runs(dirs: list, flights: list, embed: bool = True) -> list:
    """Vols demandes par le LLM, lus dans trace.jsonl : parametres choisis et verdict ecrit par le code.

    Chaque vol qui a tourne est aussi ajoute a `flights`, avec sa trajectoire, pour etre rejoue dans la scene 3D."""
    out = []
    for label, judge, d in dirs:
        trace = Path(d) / "trace.jsonl"
        if not trace.exists():
            continue
        rows, lowest, done = [], None, False
        for line in trace.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            done = done or rec["type"] in ("final", "stop")
            if rec["type"] != "tool":
                continue
            if rec["name"] == "report":
                lowest = rec["result"].get("lowest_voltage_crossed_intact_V")
            if rec["name"] != "fly":
                continue
            a, r = json.loads(rec["args"]), rec["result"]
            row = {"voltage_V": a.get("voltage_V"), "capped": a.get("capped", True), "flexible": a.get("flexible", False),
                   "ps": round(a.get("steps", 8000) / 1000, 1), "crossings": r.get("crossings"), "idx": None,
                   "verdict": r.get("verdict") if r.get("ok") else "refused by the code: threshold not computed"}
            dump = Path(d) / f"{r.get('flight')}.lammpstrj"
            if embed and r.get("ok") and dump.exists():
                steps, xyz, q, types = read_dump(dump)
                run = run_from_dump(steps, xyz, q, types, a)
                tubes = ("capped" if row["capped"] else "open") + ", " + ("flexible" if row["flexible"] else "rigid")
                row["idx"] = len(flights)
                n = len([x for x in rows if x["idx"] is not None]) + 1
                flights.append({"label": f"LLM {label.lower()}, flight {n}: {row['voltage_V']:g} V, {tubes}",
                                "verdict": row["verdict"], "run": run,
                                "traj": traj(steps, xyz, q, types, not row["flexible"],
                                             AGENT_FRAMES_FLEX if row["flexible"] else AGENT_FRAMES),
                                "crossings": crossings(run), "period_ps": None})
            rows.append(row)
        out.append({"label": label, "judge": judge, "flights": rows, "lowest_V": lowest, "finished": done})
    return out


def build(workdir: Path, key: str | None, long_key: str | None = None, all_flights: bool = False) -> dict:
    res = json.loads((workdir / "results.json").read_text())
    # par defaut : le plus long des vols flexibles a calottes
    key = key or max((k for k in res if k.startswith("fly_capped_flex")), key=lambda k: res[k].get("simulated_ps", 0))
    run = res[key]
    steps, xyz, q, types = read_dump(workdir / f"{key}.lammpstrj")
    cross = crossings(run)
    thr = res.get("threshold_capped" if run.get("capped") else "threshold", {})
    # traversees "dans l'axe" : celles qui partent avant que le C60 s'ecarte de plus de 2 A de l'axe ; ensuite il
    # rebondit sur les bords des bouches
    t_frames = np.array(steps) * nt.DT_PS
    off = np.hypot(*xyz[:, types == 3, :].mean(axis=1)[:, :2].T)
    t_miss = float(t_frames[np.argmax(off > 2.0)]) if (off > 2.0).any() else float("inf")
    for c in cross:                                              # tout ce qui part apres la premiere sortie d'axe est un rebond
        c["on_axis"] = bool(c["release_ps"] < t_miss)
    shown, shown_cross, n_f = run, cross, len(steps)
    if not all_flights:
        # animation = l'oscillation seule : on coupe a l'arrivee de la derniere traversee dans l'axe qui ramene le C60
        # dans son tube de depart, et la page la joue en boucle. La suite (derive, sortie) reste dans "It does not last".
        home = "right" if run["z_A"][0] > 0 else "left"
        back = [c["arrival_ps"] for c in cross if c["on_axis"] and c["to"] == home]
        if back:
            clip = back[-1]
            n_s = int(np.searchsorted(np.array(run["time_ps"]), clip, side="right"))
            n_f = int(np.searchsorted(t_frames, clip, side="right"))
            shown = dict(run, clip_ps=clip, **{k: run[k][:n_s] for k in ("time_ps", "z_A", "q_ball_e", "link")})
            shown_cross = [c for c in cross if c["arrival_ps"] <= clip]
    flights = [{"label": f"Run by hand, not by the LLM: {run['voltage_V']:g} V, capped, flexible", "verdict": None, "run": shown,
                "traj": traj(steps[:n_f], xyz[:n_f], q[:n_f], types, False, None), "crossings": shown_cross,
                "period_ps": period(cross)}]
    agent = agent_runs(AGENT_DIRS, flights, embed=all_flights)
    lk = long_key or key
    steps_l, ball = steps, xyz[:, types == 3, :].mean(axis=1)
    if lk != key:
        steps_l, lx, _, lt = read_dump(workdir / f"{lk}.lammpstrj")
        ball = lx[:, lt == 3, :].mean(axis=1)
    lrun = res[lk]
    long = {"simulated_ps": lrun["simulated_ps"], "asked_ps": round(lrun["steps"] * nt.DT_PS, 1), "voltage_V": lrun["voltage_V"],
            "crossings": crossings(lrun), "t_ps": [round(s * nt.DT_PS, 2) for s in steps_l],
            "off_axis_A": np.hypot(ball[:, 0], ball[:, 1]).round(2).tolist()}
    # histoire jouee par l'animation : TOUS les vols de l'agent, run apres run, dans l'ordre ou il les a demandes ;
    # le vol fait a la main (indice 0) ne vient qu'en tout dernier
    # (option --all-flights ; par defaut l'animation ne joue que l'oscillation du vol de reference)
    story = [f["idx"] for r in agent for f in r["flights"] if f["idx"] is not None] + [0]
    story_run = None
    return {"key": key, "story": story, "story_run": story_run, "well": res.get("well", {}), "threshold": {k: v for k, v in thr.items() if k != "scan"},
            "warnings": res.get("warnings", []), "bounds": fs.BOUNDS, "contact_A": fs.CONTACT_A,
            "flights": flights, "long": long, "agent": agent}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("workdir")
    ap.add_argument("--run", default=None, help="cle du vol de reference dans results.json (defaut : le plus long vol flexible a calottes)")
    ap.add_argument("--long", default=None, help="cle d'un autre vol pour le bloc 'It does not last' (defaut : le vol de reference)")
    ap.add_argument("--all-flights", action="store_true",
                    help="l'animation rejoue aussi tous les vols de l'agent, puis le vol de reference en entier (page lourde)")
    a = ap.parse_args()
    data = build(Path(a.workdir), a.run, a.long, a.all_flights)
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
    print(f"{docs / 'fullerene.html'} ({len(page) // 1024} ko), {len(data['flights'])} vols dont "
          f"{len(data['flights']) - 1} de l'agent, vol de reference : {data['flights'][0]['traj']['frames']} images")
