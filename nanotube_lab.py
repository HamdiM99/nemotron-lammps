#!/usr/bin/env python3
"""
nanotube_lab.py - navettes de nanotubes tete-beche : proprietes electromecaniques, taille reduite.

Reprend, a l'echelle d'un PC, le dispositif de M. Hamdi et al., "Rotary Nanomotors Based on Head-to-head
Nanotube Shuttles", ICRA 2010 : deux nanotubes a double paroi alignes, dont les coeurs sont attires l'un
vers l'autre par une tension. Meme principe que graphene_agent : les parametres sont bornes, c'est le code
qui ecrit l'input LAMMPS, et chaque resultat passe un controle de plausibilite.

Ce que le module calcule (pipeline deterministe, sans LLM) :
  static   barrieres de glissement et de rotation entre parois, force de rappel telescopique (AIREBO + LJ)
  charges  distribution de charge d'un tube porte a un potentiel, capacite (methode des moments, charges gaussiennes)
  force    force electrostatique entre les deux coeurs et tension de seuil, en fonction de l'ecart
  osc      oscillateur telescopique en dynamique moleculaire, frequence comparee a l'estimation analytique
  close    fermeture des deux navettes sous tension, glissement et rotation des coeurs
  all      tout, puis ecrit <workdir>/results.json

ATTENTION, ce n'est pas une reproduction de l'article : tubes (5,5)@(10,10) de 5 nm au lieu de 10 parois
de 200 nm, paroi externe figee, charges fixees pendant la dynamique.

Dependances : LAMMPS (lmp) avec CH.airebo, numpy.
"""
import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

import graphene_agent as ga

ACC = 1.42                 # distance C-C (A)
AY = math.sqrt(3) * ACC    # periode axiale d'un tube armchair (A)
K_COULOMB = 14.399645      # e2/(4 pi eps0) en eV.A
R_GAUSS = 0.6862           # largeur de la charge gaussienne du carbone (A), modele charge-dipole de Mayer
EV_A_TO_NN = 1.602177      # 1 eV/A = 1,602 nN
EV_A2_TO_J_M2 = 16.02177   # 1 eV/A2 = 16,02 J/m2
E_TO_AF_PER_V = 0.1602177  # 1 e/V = 0,1602 aF
AMU_KG = 1.660539e-27
MASS_C = 12.011
DT_PS = 0.001              # pas de temps de la dynamique (1 fs, carbone sans hydrogene)
THREADS = 8                # fils OpenMP pour la dynamique

BOUNDS = {
    "n_inner": (4, 8),            # tube interne (n,n) ; l'externe est (n+5,n+5), ecart entre parois 3,39 A
    "cells": (8, 40),             # longueur en mailles de 2,46 A
    "gap_A": (8.0, 60.0),         # ecart entre les deux navettes
    "voltage_V": (0.5, 30.0),
    "temperature_K": (1.0, 600.0),
    "steps": (1000, 400_000),
}
MAX_ATOMS = 12_000
# Plages de plausibilite : un calcul qui se termine peut etre faux.
PLAUSIBLE = {
    "interlayer_spacing_A": (3.2, 3.6),
    "adhesion_J_m2": (0.10, 0.60),          # energie d'adhesion entre parois graphitiques
    "charge_per_atom_e_at_6V": (1e-3, 0.1), # l'article donne 5 a 34 x 10^-3 e pour un tube de 5 nm a 6 V
    "osc_freq_ratio": (0.5, 2.0),           # frequence mesuree / estimation analytique
}


def check(name: str, value, errors: list):
    lo, hi = BOUNDS[name]
    if not isinstance(value, (int, float)) or not (lo <= value <= hi):
        errors.append(f"{name}={value} hors bornes [{lo}, {hi}]")


def radius(n: int) -> float:
    return 3 * ACC * n / (2 * math.pi)


# --------------------------------------------------------------------------- #
# Geometrie
# --------------------------------------------------------------------------- #
def armchair_tube(n: int, cells: int, z0: float = 0.0) -> np.ndarray:
    """Tube armchair (n,n) d'axe z : feuille de graphene enroulee selon la direction armchair."""
    ax, r = 3 * ACC, radius(n)
    basis = [(0.0, 0.0), (ACC, 0.0), (1.5 * ACC, AY / 2), (2.5 * ACC, AY / 2)]
    pts = []
    for j in range(cells):
        for i in range(n):
            for bx, by in basis:
                th = 2 * math.pi * (i * ax + bx) / (n * ax)
                pts.append((r * math.cos(th), r * math.sin(th), z0 + j * AY + by))
    return np.array(pts)


def shuttle(n: int, cells: int, z0: float = 0.0, shift: float = 0.0) -> tuple:
    """Navette = coeur (n,n) dans une paroi (n+5,n+5) ; shift = sortie du coeur selon +z."""
    return armchair_tube(n, cells, z0 + shift), armchair_tube(n + 5, cells, z0)


def head_to_head(n: int, cells: int, gap: float, protrusion: float = 0.0) -> list:
    """[coeur gauche, paroi gauche, coeur droit, paroi droite] ; les coeurs avancent chacun de `protrusion`."""
    length = (cells - 0.5) * AY
    inner_l, outer_l = shuttle(n, cells, -gap / 2 - length, protrusion)
    inner_r, outer_r = shuttle(n, cells, gap / 2, -protrusion)
    return [inner_l, outer_l, inner_r, outer_r]


def write_data(path: Path, groups: list, zbox: tuple, charges: list | None = None):
    n_tot = sum(len(g) for g in groups)
    rmax = max(float(np.abs(g[:, :2]).max()) for g in groups) + 15.0
    lines = ["nanotubes (genere par nanotube_lab)", "", f"{n_tot} atoms", f"{len(groups)} atom types", "",
             f"{-rmax:.4f} {rmax:.4f} xlo xhi", f"{-rmax:.4f} {rmax:.4f} ylo yhi", f"{zbox[0]:.6f} {zbox[1]:.6f} zlo zhi",
             "", "Masses", ""] + [f"{t + 1} {MASS_C}" for t in range(len(groups))]
    lines += ["", "Atoms # " + ("charge" if charges is not None else "atomic"), ""]
    k = 0
    for t, g in enumerate(groups):
        for a, (x, y, z) in enumerate(g):
            k += 1
            q = f" {charges[t][a]:.8f}" if charges is not None else ""
            lines.append(f"{k} {t + 1}{q} {x:.6f} {y:.6f} {z:.6f}")
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# LAMMPS
# --------------------------------------------------------------------------- #
def potential() -> str:
    p = ga.find_potential("CH.airebo")
    if p is None:
        sys.exit("CH.airebo introuvable (definir LAMMPS_POTENTIALS)")
    return p.as_posix()


def lammps(workdir: Path, name: str, text: str, timeout: int = 3600, threads: int = 0) -> str:
    lmp = shutil.which("lmp") or shutil.which("lmp_serial") or shutil.which("lmp_mpi")
    if lmp is None:
        sys.exit("executable LAMMPS (lmp) introuvable dans le PATH")
    (workdir / f"in.{name}.lammps").write_text(text)
    cmd = [lmp, "-in", f"in.{name}.lammps", "-log", f"log.{name}.lammps", "-screen", "none"]
    if threads:                                    # styles /omp (airebo/omp, coul/cut/omp) pour la dynamique
        cmd += ["-sf", "omp", "-pk", "omp", str(threads)]
    proc = subprocess.run(cmd,
                          cwd=workdir, capture_output=True, text=True, timeout=timeout)
    log = (workdir / f"log.{name}.lammps").read_text(errors="replace")
    errors = [l for l in (log + proc.stderr).splitlines() if l.startswith("ERROR")]
    if proc.returncode or errors:
        raise RuntimeError(f"LAMMPS ({name}) : {errors or proc.returncode}\n" + "\n".join(log.splitlines()[-8:]))
    return log


HEADER = """# genere par nanotube_lab - ne pas editer a la main
units metal
atom_style {style}
boundary f f {zb}
read_data {name}.data
"""
# AIREBO avec le terme de Lennard-Jones ACTIF (1) : c'est lui qui porte l'interaction entre parois.
AIREBO = 'pair_style airebo 3.0 1 0\npair_coeff * * "{pot}" {cs}\n'


def scan(workdir: Path, name: str, groups: list, zbox: tuple, periodic: bool, move: str, n_steps: int) -> list:
    """Energie du systeme pour n_steps deplacements rigides successifs du coeur (type 1). Un seul lancement."""
    write_data(workdir / f"{name}.data", groups, zbox)
    text = HEADER.format(style="atomic", zb="p" if periodic else "f", name=name)
    text += AIREBO.format(pot=potential(), cs=" ".join("C" * len(groups)))
    text += f"""group inner type 1
thermo_style custom step pe
variable e equal pe
run 0
print "SCAN 0 $e"
variable i loop {n_steps}
label top
displace_atoms inner {move} units box
run 0
print "SCAN $i $e"
next i
jump SELF top
"""
    log = lammps(workdir, name, text)
    vals = [float(l.split()[2]) for l in log.splitlines() if l.startswith("SCAN ")]
    if len(vals) != n_steps + 1:
        raise RuntimeError(f"balayage {name} : {len(vals)} points au lieu de {n_steps + 1}")
    return vals


# --------------------------------------------------------------------------- #
# 1. Statique : barrieres entre parois et force de rappel
# --------------------------------------------------------------------------- #
def static(workdir: Path, n: int = 5, cells: int = 20) -> dict:
    inner, outer = shuttle(n, cells)
    n_in = len(inner)
    out = {"n_inner": n, "n_outer": n + 5, "cells": cells, "length_nm": round((cells - 0.5) * AY / 10, 3),
           "r_inner_A": round(radius(n), 3), "r_outer_A": round(radius(n + 5), 3),
           "spacing_A": round(radius(n + 5) - radius(n), 3), "atoms_inner": n_in, "atoms_outer": len(outer)}

    # tube infini (periodique en z), parois rigides : barrieres de glissement et de rotation
    per = (0.0, cells * AY)
    e_slide = scan(workdir, "slide", [inner, outer], per, True, f"move 0 0 {AY / 24:.8f}", 24)
    e_rot = scan(workdir, "rotate", [inner, outer], per, True, "rotate 0 0 0 0 0 1 1.0", 72)
    out["slide"] = {"shift_A": [round(k * AY / 24, 4) for k in range(25)],
                    "meV_per_atom": [round((e - min(e_slide)) * 1000 / n_in, 5) for e in e_slide]}
    out["rotate"] = {"angle_deg": list(range(73)),
                     "meV_per_atom": [round((e - min(e_rot)) * 1000 / n_in, 5) for e in e_rot]}
    out["slide_barrier_meV_per_atom"] = max(out["slide"]["meV_per_atom"])
    out["rotate_barrier_meV_per_atom"] = max(out["rotate"]["meV_per_atom"])

    # tubes finis : on extrait le coeur angstrom par angstrom, la pente de E(s) est la force de rappel
    n_pull = 25
    e_pull = scan(workdir, "pull", [inner, outer], (-40.0, cells * AY + 80.0), False, "move 0 0 1.0", n_pull)
    s = np.arange(n_pull + 1, dtype=float)
    rel = np.array(e_pull) - e_pull[0]
    sel = (s >= 5) & (s <= 20)
    slope = float(np.polyfit(s[sel], rel[sel], 1)[0])                      # eV/A
    r_mid = (radius(n) + radius(n + 5)) / 2
    out["pull"] = {"extraction_A": s.tolist(), "energy_eV": [round(v, 5) for v in rel]}
    out["retraction_force_nN"] = round(slope * EV_A_TO_NN, 4)
    out["adhesion_J_m2"] = round(slope / (2 * math.pi * r_mid) * EV_A2_TO_J_M2, 4)
    return out


# --------------------------------------------------------------------------- #
# 2. Electrostatique : methode des moments avec charges gaussiennes
# --------------------------------------------------------------------------- #
def interaction_matrix(pos: np.ndarray) -> np.ndarray:
    """A[i][j] en V/e : potentiel en i cree par une charge unite en j (gaussiennes de largeur R_GAUSS)."""
    d = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt((d ** 2).sum(-1))
    np.fill_diagonal(r, 1.0)
    a = K_COULOMB / r
    close = (r < 8 * R_GAUSS) & ~np.eye(len(pos), dtype=bool)             # au-dela, erf vaut 1 a 1e-8 pres
    erf = np.vectorize(math.erf)
    a[close] *= erf(r[close] / (2 * R_GAUSS))
    np.fill_diagonal(a, K_COULOMB / (math.sqrt(math.pi) * R_GAUSS))
    return a


def solve_charges(pos: np.ndarray, volts: np.ndarray) -> np.ndarray:
    """Charges (e) telles que chaque atome soit au potentiel impose (V) : conducteur a l'equilibre."""
    return np.linalg.solve(interaction_matrix(pos), volts)


def axial_force(pos: np.ndarray, q: np.ndarray, sel: np.ndarray) -> float:
    """Force coulombienne selon z (eV/A) exercee sur les atomes `sel` par tous les autres."""
    a, b = pos[sel], pos[~sel]
    d = a[:, None, :] - b[None, :, :]
    r3 = ((d ** 2).sum(-1)) ** 1.5
    return float(K_COULOMB * (q[sel][:, None] * q[~sel][None, :] * d[:, :, 2] / r3).sum())


def charges(n: int = 5, cells: int = 20, volt: float = 6.0) -> dict:
    """Tube isole (n,n) porte a `volt` : profil de charge le long de l'axe, a comparer a la Fig. 2(a) de l'article."""
    pos = armchair_tube(n, cells)
    q = solve_charges(pos, np.full(len(pos), volt))
    z = np.round(pos[:, 2], 4)
    rings = sorted(set(z.tolist()))
    profile = [float(q[z == zz].mean()) for zz in rings]
    # moyenne par maille axiale (2 anneaux, 4n atomes) : lisse l'alternance de signe entre les deux premiers
    # anneaux du bord, artefact connu d'un modele a charges ponctuelles sans dipoles
    per_cell = q.reshape(cells, 4 * n).mean(axis=1)
    return {"n": n, "cells": cells, "length_nm": round((cells - 0.5) * AY / 10, 3), "voltage_V": volt,
            "z_nm": [round(zz / 10, 4) for zz in rings], "q_e": [round(v, 6) for v in profile],
            "z_cell_nm": [round((j + 0.25) * AY / 10, 4) for j in range(cells)],
            "q_cell_e": [round(float(v), 6) for v in per_cell],
            "q_cell_min_e": round(float(per_cell.min()), 6), "q_cell_max_e": round(float(per_cell.max()), 6),
            "q_min_e": round(float(q.min()), 6), "q_max_e": round(float(q.max()), 6),
            "q_total_e": round(float(q.sum()), 4), "capacitance_aF": round(float(q.sum()) / volt * E_TO_AF_PER_V, 4),
            "end_enhancement": round(float(q.max() / np.median(q)), 3)}


def force(n: int = 5, cells: int = 20, gaps: tuple = (10.0, 15.0, 20.0, 30.0, 50.0), retraction_nN: float | None = None) -> dict:
    """Force electrostatique qui pousse un coeur vers l'autre, pour 1 V entre les navettes (F varie en V2)."""
    out = {"n": n, "cells": cells, "gaps": []}
    for gap in gaps:
        rows, travel = [], (gap - 3.4) / 2                                   # contact quand les bouts sont a 3,4 A
        for frac in np.linspace(0.0, 1.0, 9):
            e = frac * travel
            groups = head_to_head(n, cells, gap, e)
            pos = np.vstack(groups)
            sizes = np.cumsum([0] + [len(g) for g in groups])
            volts = np.where(np.arange(len(pos)) < sizes[2], 0.5, -0.5)      # gauche +V/2, droite -V/2
            q = solve_charges(pos, volts)
            sel = np.zeros(len(pos), dtype=bool)
            sel[sizes[0]:sizes[1]] = True                                    # coeur gauche
            f1 = axial_force(pos, q, sel) * EV_A_TO_NN                       # nN pour 1 V, vers +z
            rows.append({"protrusion_A": round(float(e), 3), "end_gap_A": round(float(gap - 2 * e), 3),
                         "force_nN_per_V2": round(f1, 6), "q_core_e_per_V": round(float(q[sel].sum()), 5)})
        entry = {"gap_A": gap, "scan": rows}
        if retraction_nN:
            f0 = rows[0]["force_nN_per_V2"]
            entry["threshold_V"] = round(math.sqrt(retraction_nN / f0), 2) if f0 > 0 else None
        out["gaps"].append(entry)
    return out


# --------------------------------------------------------------------------- #
# 3. Dynamique
# --------------------------------------------------------------------------- #
def read_dump(path: Path) -> tuple:
    """(pas, positions[frame][atome][xyz], types) ; atomes tries par id."""
    steps, frames, types = [], [], None
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("ITEM: TIMESTEP"):
            steps.append(int(lines[i + 1]))
            n = int(lines[i + 3])
            cols = lines[i + 8].split()[2:]
            arr = np.array([l.split() for l in lines[i + 9:i + 9 + n]], dtype=float)
            arr = arr[np.argsort(arr[:, cols.index("id")])]
            types = arr[:, cols.index("type")].astype(int)
            frames.append(arr[:, [cols.index(c) for c in ("x", "y", "z")]])
            i += 9 + n
        else:
            i += 1
    return steps, np.array(frames), types


def thermo_cols(log: str) -> dict:
    rows = ga.parse_thermo(log)[-1]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def oscillator(workdir: Path, n: int = 5, cells: int = 20, pull: float = 15.0, steps: int = 120_000,
               retraction_nN: float | None = None) -> dict:
    """Le coeur est tire de `pull` angstroms puis lache : il oscille dans la paroi externe, figee."""
    errors = []
    check("steps", steps, errors)
    if errors:
        return {"ok": False, "errors": errors}
    inner, outer = shuttle(n, cells)
    write_data(workdir / "osc.data", [inner, outer], (-80.0, cells * AY + 120.0))
    dt = DT_PS
    text = HEADER.format(style="atomic", zb="f", name="osc") + AIREBO.format(pot=potential(), cs="C C")
    text += f"""group inner type 1
group outer type 2
fix hold outer setforce 0.0 0.0 0.0
compute cm inner com
thermo_style custom step pe ke c_cm[3]
thermo 100
thermo_modify flush yes
min_style cg
minimize 1e-10 1e-10 2000 20000
displace_atoms inner move 0 0 {pull} units box
reset_timestep 0
timestep {dt}
fix move inner nve
dump traj all custom {max(1, steps // 120)} osc.lammpstrj id type x y z
dump_modify traj sort id
run {steps}
"""
    cols = thermo_cols(lammps(workdir, "osc", text, timeout=7200, threads=THREADS))
    t = cols["Step"] * dt                                                   # ps
    z = cols["c_cm[3]"] - (cols["c_cm[3]"][0] - pull)                       # deplacement par rapport a l'equilibre
    cross = [float(t[k] - z[k] * (t[k + 1] - t[k]) / (z[k + 1] - z[k]))
             for k in range(len(z) - 1) if z[k] * z[k + 1] < 0]
    out = {"ok": True, "pull_A": pull, "steps": steps, "time_ps": [round(float(v), 3) for v in t[::4]],
           "z_A": [round(float(v), 3) for v in z[::4]], "crossings_ps": [round(c, 3) for c in cross]}
    if len(cross) >= 2:
        period = 2 * float(np.mean(np.diff(cross)))
        out["period_ps"], out["frequency_GHz"] = round(period, 3), round(1000 / period, 2)
    if retraction_nN:
        mass = len(inner) * MASS_C * AMU_KG
        period = 4 * math.sqrt(2 * mass * pull * 1e-10 / (retraction_nN * 1e-9)) * 1e12   # force de rappel constante
        out["analytic_period_ps"], out["analytic_frequency_GHz"] = round(period, 3), round(1000 / period, 2)
        if "frequency_GHz" in out:
            out["freq_ratio"] = round(out["frequency_GHz"] / out["analytic_frequency_GHz"], 3)
    return out


def close_shuttles(workdir: Path, n: int = 5, cells: int = 20, gap: float = 15.0, volt: float = 10.0,
                   temperature: float = 300.0, steps: int = 80_000) -> dict:
    """Deux navettes tete-beche sous tension : les coeurs glissent l'un vers l'autre ; tournent-ils ?"""
    errors = []
    for name, v in (("gap_A", gap), ("voltage_V", volt), ("temperature_K", temperature), ("steps", steps)):
        check(name, v, errors)
    if errors:
        return {"ok": False, "errors": errors}
    groups = head_to_head(n, cells, gap)
    pos = np.vstack(groups)
    sizes = np.cumsum([0] + [len(g) for g in groups])
    q = solve_charges(pos, np.where(np.arange(len(pos)) < sizes[2], volt / 2, -volt / 2))
    qs = [q[sizes[k]:sizes[k + 1]] for k in range(4)]
    zlo, zhi = float(pos[:, 2].min()) - 60.0, float(pos[:, 2].max()) + 60.0
    write_data(workdir / "close.data", groups, (zlo, zhi), qs)
    dt, every = DT_PS, max(1, steps // 160)
    pot = potential()
    text = HEADER.format(style="charge", zb="f", name="close") + AIREBO.format(pot=pot, cs="C C C C")
    text += f"""group inner type 1 3
group outer type 2 4
group innerL type 1
group innerR type 3
fix hold outer setforce 0.0 0.0 0.0
# relaxation des coeurs SANS les charges, pour ne pas commencer a glisser pendant la minimisation
min_style cg
minimize 1e-10 1e-10 2000 20000
# puis on ajoute l'electrostatique : charges fixes, calculees pour la geometrie initiale a {volt} V
neigh_modify one 6000 page 120000
pair_style hybrid/overlay airebo 3.0 1 0 coul/cut 80.0
pair_coeff * * airebo "{pot}" C C C C
pair_coeff * * coul/cut
reset_timestep 0
timestep {dt}
velocity inner create {temperature} 9172345 mom yes rot yes dist gaussian
compute cmL innerL com
compute cmR innerR com
thermo_style custom step temp pe c_cmL[3] c_cmR[3]
thermo 100
thermo_modify flush yes
fix move inner nve
dump traj all custom {every} close.lammpstrj id type x y z
dump_modify traj sort id
run {steps}
"""
    cols = thermo_cols(lammps(workdir, "close", text, timeout=14400, threads=THREADS))
    t = cols["Step"] * dt
    steps_d, frames, types = read_dump(workdir / "close.lammpstrj")
    res = {"ok": True, "gap_A": gap, "voltage_V": volt, "temperature_K": temperature, "steps": steps,
           "time_ps": [round(float(v), 3) for v in t[::4]],
           "zL_A": [round(float(v), 3) for v in (cols["c_cmL[3]"] - cols["c_cmL[3]"][0])[::4]],
           "zR_A": [round(float(v), 3) for v in (cols["c_cmR[3]"] - cols["c_cmR[3]"][0])[::4]],
           "q_core_e": round(float(qs[0].sum()), 4), "q_min_e": round(float(q.min()), 5), "q_max_e": round(float(q.max()), 5)}
    # ecart entre les bouts des deux coeurs, et rotation de chaque coeur autour de l'axe
    end_gap, rot = [], {1: [0.0], 3: [0.0]}
    for f in frames:
        end_gap.append(float(f[types == 3][:, 2].min() - f[types == 1][:, 2].max()))
    for ty in (1, 3):
        phi = np.arctan2(frames[:, types == ty, 1], frames[:, types == ty, 0])
        d = np.diff(phi, axis=0)
        d = (d + math.pi) % (2 * math.pi) - math.pi
        rot[ty] = [0.0] + np.degrees(np.cumsum(d.mean(axis=1))).round(3).tolist()
    tf = [round(s * dt, 3) for s in steps_d]
    res.update({"frame_time_ps": tf, "end_gap_A": [round(v, 3) for v in end_gap],
                "rotation_L_deg": rot[1], "rotation_R_deg": rot[3]})
    contact = next((k for k, g in enumerate(end_gap) if g < 3.6), None)
    res["contact_time_ps"] = tf[contact] if contact is not None else None
    res["min_end_gap_A"] = round(min(end_gap), 3)
    return res


# --------------------------------------------------------------------------- #
# Controles de plausibilite
# --------------------------------------------------------------------------- #
def plausibility(r: dict) -> list:
    warn = []

    def rng(key, value, label):
        lo, hi = PLAUSIBLE[key]
        if value is not None and not (lo <= value <= hi):
            warn.append(f"{label} = {value} hors plage [{lo}, {hi}]")

    if "static" in r:
        rng("interlayer_spacing_A", r["static"]["spacing_A"], "ecart entre parois (A)")
        rng("adhesion_J_m2", r["static"]["adhesion_J_m2"], "energie d'adhesion (J/m2)")
    if "charges" in r:
        scale = 6.0 / r["charges"]["voltage_V"]
        rng("charge_per_atom_e_at_6V", r["charges"]["q_max_e"] * scale, "charge maximale par atome a 6 V (e)")
    if r.get("oscillator", {}).get("freq_ratio") is not None:
        rng("osc_freq_ratio", r["oscillator"]["freq_ratio"], "frequence mesuree / analytique")
    return warn


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Navettes de nanotubes tete-beche (taille reduite)")
    ap.add_argument("task", choices=["static", "charges", "force", "osc", "close", "all"])
    ap.add_argument("--workdir", default="runs_nt")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--cells", type=int, default=20)
    ap.add_argument("--gap", type=float, default=15.0)
    ap.add_argument("--volt", type=float, default=None, help="tension pour 'close' (defaut : 1,5 x le seuil)")
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args()
    errs = []
    check("n_inner", a.n, errs)
    check("cells", a.cells, errs)
    if 4 * a.cells * (4 * a.n + 10) > MAX_ATOMS:
        errs.append(f"trop d'atomes (budget {MAX_ATOMS})")
    if errs:
        sys.exit(" ; ".join(errs))
    wd = Path(a.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    out_file = wd / "results.json"
    res = json.loads(out_file.read_text()) if out_file.exists() else {}

    def save():
        res["warnings"] = plausibility(res)
        out_file.write_text(json.dumps(res, indent=1))

    if a.task in ("static", "all"):
        res["static"] = static(wd, a.n, a.cells)
        save()
        print("static :", {k: v for k, v in res["static"].items() if not isinstance(v, dict)})
    f_ret = res.get("static", {}).get("retraction_force_nN")
    if a.task in ("charges", "all"):
        res["charges"] = charges(a.n, a.cells, 6.0)
        save()
        print("charges :", {k: v for k, v in res["charges"].items() if not isinstance(v, list)})
    if a.task in ("force", "all"):
        res["force"] = force(a.n, a.cells, retraction_nN=f_ret)
        save()
        for g in res["force"]["gaps"]:
            print(f"force : ecart {g['gap_A']} A, F = {g['scan'][0]['force_nN_per_V2']} nN/V2 au depart, "
                  f"{g['scan'][-1]['force_nN_per_V2']} au contact, seuil {g.get('threshold_V')} V")
    if a.task in ("osc", "all"):
        res["oscillator"] = oscillator(wd, a.n, a.cells, steps=a.steps or 120_000, retraction_nN=f_ret)
        save()
        print("oscillateur :", {k: v for k, v in res["oscillator"].items() if not isinstance(v, list)})
    if a.task in ("close", "all"):
        thr = next((g.get("threshold_V") for g in res.get("force", {}).get("gaps", []) if g["gap_A"] == a.gap), None)
        volt = a.volt or (round(1.5 * thr, 1) if thr else 10.0)
        res["close"] = close_shuttles(wd, a.n, a.cells, a.gap, volt, steps=a.steps or 80_000)
        if thr:
            res["close"]["threshold_V"] = thr
        save()
        print("fermeture :", {k: v for k, v in res["close"].items() if not isinstance(v, list)})
    save()
    print("avertissements :", res["warnings"] or "aucun")
