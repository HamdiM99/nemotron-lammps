#!/usr/bin/env python3
"""
fullerene_shuttle.py - navette a fullerene : un C60 passe d'un nanotube a l'autre sous tension.

Deux nanotubes (10,10) ouverts, face a face, portes a +V/2 et -V/2. Un C60 loge pres de la bouche du tube
de droite prend la charge de ce tube, est attire par le tube d'en face, traverse l'ecart et y entre.
Meme principe que nanotube_lab : parametres bornes, input LAMMPS ecrit par le code, controle de plausibilite.

Ce que le module calcule (deterministe, sans LLM) :
  well       puits de van der Waals du C60 dans le tube, force de retenue a la bouche (AIREBO + LJ)
  threshold  charge du C60 et tension de seuil d'ejection (methode des moments), en fonction de la position
  fly        dynamique : le C60 quitte un tube et entre dans l'autre ; sa charge est RECALCULEE en vol
  all        tout, puis ecrit <workdir>/results.json

Modele de charge pendant le vol (toutes les `update` iterations) :
  - C60 a moins de CONTACT_A d'un tube : il est au potentiel de ce tube (contact electrique) ;
  - sinon : conducteur isole, il garde sa charge totale, qui se redistribue sur ses 60 atomes ;
  - les charges des tubes sont recalculees a chaque fois, tubes maintenus a +V/2 et -V/2.

ATTENTION : tubes rigides et figes (pas de dissipation), bouts ouverts sans calotte, pas d'emission de champ,
contact electrique decide par un simple seuil de distance.
Artefact connu : au fond d'un tube, un conducteur au meme potentiel devrait porter une charge nulle (cage de
Faraday). Avec des charges gaussiennes, le C60 y garde une petite charge de signe OPPOSE a celle du tube
(environ 0,01 e/V, contre 0,08 e/V a la sortie). Elle ne pilote pas l'ejection, qui se joue a la bouche.

Dependances : LAMMPS (executable lmp et module Python lammps) avec CH.airebo, numpy.
"""
import argparse
import ctypes
import json
import math
import sys
from itertools import product
from pathlib import Path

import numpy as np

import nanotube_lab as nt

TUBE_N = 10                # tubes (10,10), rayon 6,78 A : le C60 y loge a 3,2 A de la paroi
C60_BOND = 1.43            # longueur de liaison moyenne du C60 (A)
CONTACT_A = 4.0            # en dessous de cette distance atome-atome, le C60 est en contact electrique avec le tube
COUL_CUT = 120.0           # coupure coulombienne (A), plus grande que le systeme : aucune troncature
MASS_C60_KG = 60 * nt.MASS_C * nt.AMU_KG

BOUNDS = {
    "cells": (8, 30),              # longueur de chaque tube en mailles de 2,46 A
    "gap_A": (16.0, 40.0),         # ecart entre les deux bouches ; >= 16 A pour que le C60 ne touche pas les deux tubes
    "depth_A": (0.0, 15.0),        # enfoncement initial du centre du C60 derriere la bouche du tube de droite
    "voltage_V": (0.5, 30.0),
    "temperature_K": (0.0, 600.0),
    "steps": (1000, 200_000),
    "update": (10, 1000),          # iterations entre deux recalculs des charges
}
# Plages de plausibilite, fixees avant le premier calcul.
PLAUSIBLE = {
    "wall_gap_A": (3.0, 3.6),            # distance C60-paroi dans un (10,10)
    "binding_eV": (2.0, 5.0),            # energie d'encapsulation du C60 dans un (10,10) ; litterature : environ 3 a 4 eV
    "retention_nN": (0.1, 3.0),          # force de retenue maximale a la bouche
    "charge_drift_e": (0.0, 1e-6),       # derive de la charge du C60 pendant le vol libre (doit etre conservee)
    "cap_bond_A": (1.33, 1.55),          # longueurs de liaison dans la calotte relaxee
}


def check(name: str, value, errors: list):
    lo, hi = BOUNDS[name]
    if not isinstance(value, (int, float)) or not (lo <= value <= hi):
        errors.append(f"{name}={value} hors bornes [{lo}, {hi}]")


# --------------------------------------------------------------------------- #
# Geometrie
# --------------------------------------------------------------------------- #
def c60(center=(0.0, 0.0, 0.0)) -> np.ndarray:
    """C60 : icosaedre tronque, 60 sommets, liaisons de C60_BOND."""
    phi = (1 + math.sqrt(5)) / 2
    pts = set()
    for base in ((0, 1, 3 * phi), (1, 2 + phi, 2 * phi), (phi, 2, 2 * phi + 1)):
        for signs in product((1, -1), repeat=3):
            v = [s * c for s, c in zip(signs, base)]
            for k in range(3):                                   # permutations paires = permutations circulaires
                pts.add(tuple(round(v[(i + k) % 3], 9) + 0.0 for i in range(3)))
    return np.array(sorted(pts)) * (C60_BOND / 2) + np.array(center)


def c60_radius() -> float:
    phi = (1 + math.sqrt(5)) / 2
    return math.sqrt(9 * phi ** 2 + 1) * C60_BOND / 2


def tube_length(cells: int) -> float:
    return (cells - 0.5) * nt.AY


def setup(cells: int = 12, gap: float = 20.0, depth: float = 3.0, tube: np.ndarray | None = None) -> list:
    """[tube gauche, tube droit, C60]. Bouches en z = -gap/2 et +gap/2 ; le C60 est dans le tube de droite.

    tube : coordonnees d'un tube dans son repere propre (bouche en z = 0, corps vers +z), par exemple un tube
    a calotte ; par defaut un (10,10) ouvert. Le tube de gauche est l'image miroir du tube de droite."""
    if tube is None:
        tube = nt.armchair_tube(TUBE_N, cells, 0.0)
    right = tube + np.array([0.0, 0.0, gap / 2])
    left = tube * np.array([1.0, 1.0, -1.0]) + np.array([0.0, 0.0, -gap / 2])
    return [left, right, c60((0.0, 0.0, gap / 2 + depth))]


# --------------------------------------------------------------------------- #
# Calotte : un demi-C240 ferme un tube (10,10), comme un demi-C60 ferme un (5,5)
# --------------------------------------------------------------------------- #
def c240_unit() -> np.ndarray:
    """C240 icosaedrique sur la sphere unite, un axe d'ordre 5 selon z. Dual du polyedre geodesique (2,2)."""
    phi = (1 + math.sqrt(5)) / 2
    v = []
    for a, b in product((1, -1), repeat=2):
        for k in range(3):
            p = [0.0, a * 1.0, b * phi]
            v.append([p[(i + k) % 3] for i in range(3)])
    v = np.array(v) / math.sqrt(1 + phi ** 2)
    edge = min(np.linalg.norm(v[0] - v[j]) for j in range(1, 12))
    near = lambda i, j: abs(np.linalg.norm(v[i] - v[j]) - edge) < 1e-6
    faces = [(i, j, k) for i in range(12) for j in range(i + 1, 12) for k in range(j + 1, 12)
             if near(i, j) and near(j, k) and near(i, k)]
    unit = lambda p: p / np.linalg.norm(p)
    centre = [unit(v[list(f)].mean(axis=0)) for f in faces]
    atoms = []
    for i in range(12):                                          # dodecaedre pentakis : 60 triangles (sommet, 2 centres de faces voisines)
        mine = [f for f in range(20) if i in faces[f]]
        for a in range(5):
            for b in range(a + 1, 5):
                if len(set(faces[mine[a]]) & set(faces[mine[b]])) == 2:
                    p, q, r = v[i], centre[mine[a]], centre[mine[b]]
                    pq, qr, rp = unit(p + q), unit(q + r), unit(r + p)
                    for tri in ((p, pq, rp), (q, qr, pq), (r, rp, qr), (pq, qr, rp)):      # chaque triangle coupe en 4
                        atoms.append(unit(sum(tri)))
    atoms = np.array(atoms)
    # amene le sommet 0 de l'icosaedre (axe d'ordre 5) sur +z
    z = v[0]
    x = unit(np.cross([0.0, 0.0, 1.0], z) if abs(z[2]) < 0.9 else np.cross([1.0, 0.0, 0.0], z))
    return atoms @ np.array([x, np.cross(z, x), z]).T


def neighbours(pos: np.ndarray, k: int = 3) -> np.ndarray:
    d = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
    return np.argsort(d, axis=1)[:, 1:k + 1]


def capped_tube(cells: int = 12, stretch: float = 1.15) -> np.ndarray:
    """Tube (10,10) dans son repere propre (bouche ouverte en z = 0), ferme a l'autre bout par un demi-C240.

    Le C240 a, autour de son equateur, un anneau armchair de 20 atomes qui le separe en deux moities de 110 :
    le dernier anneau du tube prend la place de cet anneau, et une moitie devient la calotte (110 atomes).
    Geometrie pre-relaxee par des ressorts sur le graphe des liaisons ; a finir avec relax_cap() (AIREBO)."""
    tube = nt.armchair_tube(TUBE_N, cells, 0.0)
    z_end = float(tube[:, 2].max())
    ring = np.where(np.isclose(tube[:, 2], z_end))[0]             # dernier anneau : 20 atomes a une liaison pendante
    ball = c240_unit()
    nb = neighbours(ball)
    belt = np.where(np.abs(ball[:, 2]) < 0.05)[0]                 # anneau equatorial, remplace par l'anneau du tube
    up = np.where(ball[:, 2] > 0.05)[0]
    if len(belt) != 2 * TUBE_N or len(up) != 110:
        raise RuntimeError(f"C240 inattendu : anneau de {len(belt)} atomes, calotte de {len(up)}")
    # rotation de la calotte qui amene l'anneau equatorial en face de l'anneau du tube, puis appariement un a un
    ring_ang = np.arctan2(tube[ring, 1], tube[ring, 0])
    belt_ang = np.arctan2(ball[belt, 1], ball[belt, 0])
    wrap = lambda d: (d + math.pi) % (2 * math.pi) - math.pi

    def mismatch(rot):
        return float(np.abs(wrap(belt_ang[:, None] + rot - ring_ang[None, :])).min(axis=1).sum())
    rot = min(np.radians(np.arange(0.0, 72.0, 0.05)), key=mismatch)
    partner = np.abs(wrap(belt_ang[:, None] + rot - ring_ang[None, :])).argmin(axis=1)
    if len(set(partner.tolist())) != len(belt):
        raise RuntimeError("raccord calotte-tube : appariement des anneaux non bijectif")
    to_tube = {int(b): int(ring[p]) for b, p in zip(belt, partner)}
    index = {int(a): len(tube) + k for k, a in enumerate(up)}     # numero de chaque atome de la calotte dans le tube ferme
    index.update(to_tube)
    c, s = math.cos(rot), math.sin(rot)
    scale = nt.radius(TUBE_N) / float(np.hypot(ball[belt, 0], ball[belt, 1]).mean())
    cap = ball[up] @ np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    cap[:, :2] *= scale
    cap[:, 2] = z_end + cap[:, 2] * scale * stretch
    pos = np.vstack([tube, cap])
    # liaisons de la calotte (y compris vers l'anneau du tube) et seconds voisins, pour la pre-relaxation
    bonds = sorted({tuple(sorted((index[int(a)], index[int(j)]))) for a in up for j in nb[a]})
    adj = {}
    for i, j in bonds:
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)
    for i in ring:                                                # l'anneau du tube garde aussi ses voisins du tube
        d = np.linalg.norm(tube - tube[i], axis=1)
        adj[int(i)] += [int(j) for j in np.where((d > 0.1) & (d < 1.6))[0]]
    seconds = sorted({tuple(sorted((a, b))) for k, v in adj.items() for a in v for b in v if a < b})
    springs = [(np.array(p), r0, k) for p, r0, k in ((bonds, nt.ACC, 1.0), (seconds, math.sqrt(3) * nt.ACC, 0.3))]
    free = np.arange(len(tube), len(pos))
    for _ in range(4000):                                         # descente de gradient, seuls les 110 atomes de la calotte bougent
        force = np.zeros_like(pos)
        for pairs, r0, k in springs:
            d = pos[pairs[:, 1]] - pos[pairs[:, 0]]
            r = np.linalg.norm(d, axis=1)
            f = (k * (r - r0) / r)[:, None] * d
            np.add.at(force, pairs[:, 0], f)
            np.add.at(force, pairs[:, 1], -f)
        pos[free] += 0.15 * force[free]
    return pos


def cap_quality(tube: np.ndarray) -> dict:
    """Coordination et longueurs de liaison d'un tube a calotte : seuls les 20 atomes de la bouche ont 2 voisins."""
    d = np.linalg.norm(tube[:, None] - tube[None, :], axis=-1)
    bonded = (d > 0.1) & (d < 1.75)
    coord = bonded.sum(axis=1)
    return {"atoms": len(tube), "two_neighbours": int((coord == 2).sum()), "three_neighbours": int((coord == 3).sum()),
            "other": int(((coord != 2) & (coord != 3)).sum()),
            "bond_min_A": round(float(d[bonded].min()), 3), "bond_max_A": round(float(d[bonded].max()), 3)}


def relax_cap(workdir: Path, cells: int = 12) -> np.ndarray:
    """Relaxe la calotte et les deux mailles voisines (AIREBO), le reste du tube fige. Resultat garde en cache."""
    cache = workdir / f"capped_{cells}.json"
    if cache.exists():
        return np.array(json.loads(cache.read_text())["xyz"])
    tube = capped_tube(cells)
    free = tube[:, 2] > tube_length(cells) - 2 * nt.AY
    name = f"cap_{cells}"
    nt.write_data(workdir / f"{name}.data", [tube[~free], tube[free]], (-40.0, float(tube[:, 2].max()) + 40.0))
    text = nt.HEADER.format(style="atomic", zb="f", name=name) + nt.AIREBO.format(pot=nt.potential(), cs="C C")
    text += f"""group body type 1
fix hold body setforce 0.0 0.0 0.0
min_style cg
minimize 1e-12 1e-12 20000 200000
write_dump all custom {name}.dump id type x y z modify sort id
"""
    nt.lammps(workdir, name, text)
    _, frames, _ = nt.read_dump(workdir / f"{name}.dump")
    relaxed = np.empty_like(tube)
    relaxed[~free], relaxed[free] = frames[-1][:int((~free).sum())], frames[-1][int((~free).sum()):]
    q = cap_quality(relaxed)
    lo, hi = PLAUSIBLE["cap_bond_A"]
    if q["other"] or q["two_neighbours"] != 2 * TUBE_N or not (lo <= q["bond_min_A"] and q["bond_max_A"] <= hi):
        raise RuntimeError(f"calotte mal formee apres relaxation : {q}")
    cache.write_text(json.dumps({"quality": q, "xyz": relaxed.round(6).tolist()}))
    return relaxed


# --------------------------------------------------------------------------- #
# Electrostatique : conducteurs a potentiel impose, C60 relie ou isole
# --------------------------------------------------------------------------- #
def block(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Potentiel (V/e) cree en a par des charges gaussiennes unite en b ; meme noyau que nt.interaction_matrix."""
    r = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    out = nt.K_COULOMB / np.maximum(r, 1e-9)
    close = r < 8 * nt.R_GAUSS
    rc = r[close]
    erf = np.vectorize(math.erf, otypes=[float])
    self_term = nt.K_COULOMB / (math.sqrt(math.pi) * nt.R_GAUSS)
    out[close] = np.where(rc > 1e-9, out[close] * erf(rc / (2 * nt.R_GAUSS)), self_term)
    return out


def link_of(ball: np.ndarray, left: np.ndarray, right: np.ndarray) -> str | None:
    """'L' ou 'R' si le C60 est en contact electrique avec ce tube, None s'il est isole."""
    def dmin(tube):
        return float(np.sqrt(((ball[:, None, :] - tube[None, :, :]) ** 2).sum(-1)).min())
    dl, dr = dmin(left), dmin(right)
    if min(dl, dr) >= CONTACT_A:
        return None
    return "L" if dl < dr else "R"


def solve_state(a: np.ndarray, n_left: int, n_right: int, volt: float, link: str | None, q_ball: float = 0.0) -> tuple:
    """Charges (e) de tous les atomes et potentiel du C60 (V). Ordre des atomes : gauche, droite, C60.

    link 'L'/'R' : le C60 est au potentiel du tube ; None : C60 isole de charge totale q_ball, equipotentiel."""
    n = a.shape[0]
    nb = n - n_left - n_right
    v = np.empty(n)
    v[:n_left], v[n_left:n_left + n_right] = volt / 2, -volt / 2
    if link:
        v[-nb:] = volt / 2 if link == "L" else -volt / 2
        return np.linalg.solve(a, v), float(v[-1])
    v[-nb:] = 0.0
    m = np.zeros((n + 1, n + 1))
    m[:n, :n] = a
    m[n - nb:n, n] = -1.0                       # potentiel inconnu du C60, le meme sur ses 60 atomes
    m[n, n - nb:n] = 1.0                        # charge totale imposee
    sol = np.linalg.solve(m, np.append(v, q_ball))
    return sol[:n], float(sol[n])


class Electro:
    """Matrice d'interaction avec tubes fixes : seules les lignes et colonnes du C60 sont recalculees."""

    def __init__(self, left: np.ndarray, right: np.ndarray, ball: np.ndarray):
        self.left, self.right = left, right
        self.tubes = np.vstack([left, right])
        self.nt_ = len(self.tubes)
        n = self.nt_ + len(ball)
        self.a = np.empty((n, n))
        self.a[:self.nt_, :self.nt_] = block(self.tubes, self.tubes)

    def update(self, ball: np.ndarray, volt: float, q_ball: float, link: str | None = "auto",
               tubes: np.ndarray | None = None) -> tuple:
        """(charges, lien, potentiel du C60) pour la position courante du C60 (et des tubes s'ils sont flexibles)."""
        k = self.nt_
        if tubes is not None:
            self.tubes, self.left, self.right = tubes, tubes[:len(self.left)], tubes[len(self.left):]
            self.a[:k, :k] = block(tubes, tubes)
        self.a[k:, :k] = block(ball, self.tubes)
        self.a[:k, k:] = self.a[k:, :k].T
        self.a[k:, k:] = block(ball, ball)
        if link == "auto":
            link = link_of(ball, self.left, self.right)
        q, phi = solve_state(self.a, len(self.left), len(self.right), volt, link, q_ball)
        return q, link, phi


def ball_force_z(tubes: np.ndarray, ball: np.ndarray, q: np.ndarray) -> float:
    """Force coulombienne selon z (eV/A) exercee par les tubes sur le C60."""
    pos = np.vstack([tubes, ball])
    sel = np.zeros(len(pos), dtype=bool)
    sel[len(tubes):] = True
    return nt.axial_force(pos, q, sel)


# --------------------------------------------------------------------------- #
# 1. Puits de van der Waals et force de retenue
# --------------------------------------------------------------------------- #
def well(workdir: Path, cells: int = 12, step: float = 0.5, outside: float = 15.0) -> dict:
    """Energie du C60 rigide le long de l'axe d'un tube, du milieu du tube jusqu'a `outside` A hors de la bouche."""
    length = tube_length(cells)
    tube = nt.armchair_tube(TUBE_N, cells, 0.0)                 # bouche en z = 0, le C60 sort vers -z
    n_steps = int(round((length / 2 + outside) / step))
    energies = nt.scan(workdir, "well", [c60((0.0, 0.0, length / 2)), tube], (-outside - 40.0, length + 40.0),
                       False, f"move 0 0 {-step:.6f}", n_steps)
    d = np.array([-(length / 2 - k * step) for k in range(n_steps + 1)])       # position hors de la bouche (A), < 0 dedans
    e = np.array(energies) - energies[-1]                        # zero quand le C60 est loin dehors
    f_ret = np.gradient(e, d)                                    # eV/A, > 0 : le tube retient le C60
    out = {"cells": cells, "tube": f"({TUBE_N},{TUBE_N})", "length_A": round(length, 3),
           "wall_gap_A": round(nt.radius(TUBE_N) - c60_radius(), 3),
           "d_A": d.round(3).tolist(), "energy_eV": e.round(5).tolist(), "retention_eV_A": f_ret.round(5).tolist(),
           "binding_eV": round(float(-e.min()), 4),
           "retention_max_nN": round(float(f_ret.max()) * nt.EV_A_TO_NN, 4),
           "retention_max_at_A": round(float(d[int(np.argmax(f_ret))]), 2)}
    return out


# --------------------------------------------------------------------------- #
# 2. Charge du C60 et tension de seuil
# --------------------------------------------------------------------------- #
def threshold(cells: int = 12, gap: float = 20.0, depth: float = 3.0, well_res: dict | None = None, step: float = 0.5,
              tube: np.ndarray | None = None) -> dict:
    """Le C60 avance du depart jusqu'au milieu de l'ecart ; charge et force electrostatique pour 1 V (F varie en V2).

    Seuil statique : tension pour laquelle la traction electrostatique depasse la retenue de van der Waals
    en chaque point du trajet. C'est une borne haute : en dynamique, l'elan aide a franchir les points durs."""
    errors = []
    for bound, v in (("cells", cells), ("gap_A", gap), ("depth_A", depth)):
        check(bound, v, errors)
    if errors:
        return {"ok": False, "errors": errors}
    left, right, ball0 = setup(cells, gap, depth, tube)
    el = Electro(left, right, ball0)
    rows, q_ball = [], 0.0
    for k in range(int(round((depth + gap / 2) / step)) + 1):
        d = -depth + k * step                                    # position du centre hors de la bouche de droite
        ball = ball0 + np.array([0.0, 0.0, -(k * step)])
        q, link, phi = el.update(ball, 1.0, q_ball)
        q_ball = float(q[el.nt_:].sum())                         # garde en memoire : conservee une fois le contact perdu
        f_pull = -ball_force_z(el.tubes, ball, q)                # eV/A par V2, > 0 : vers le tube de gauche
        rows.append({"d_A": round(d, 3), "link": link, "q_ball_e_per_V": round(q_ball, 6),
                     "pull_nN_per_V2": round(f_pull * nt.EV_A_TO_NN, 6)})
    out = {"ok": True, "cells": cells, "gap_A": gap, "depth_A": depth, "capped": tube is not None, "scan": rows,
           "q_ball_start_e_per_V": rows[0]["q_ball_e_per_V"],
           "q_ball_flight_e_per_V": next((r["q_ball_e_per_V"] for r in rows if r["link"] is None), None),
           "release_at_A": next((r["d_A"] for r in rows if r["link"] is None), None)}
    if well_res:
        dw, fw = np.array(well_res["d_A"]), np.array(well_res["retention_eV_A"]) * nt.EV_A_TO_NN
        need = []
        for r in rows:
            ret = float(np.interp(r["d_A"], dw, fw))             # retenue de van der Waals au meme point (nN)
            r["retention_nN"] = round(ret, 5)
            if ret > 0:
                need.append(math.sqrt(ret / r["pull_nN_per_V2"]) if r["pull_nN_per_V2"] > 0 else math.inf)
        out["threshold_V"] = round(max(need), 2) if need and math.isfinite(max(need)) else None
    return out


# --------------------------------------------------------------------------- #
# 3. Dynamique avec charges recalculees
# --------------------------------------------------------------------------- #
def location(z: float, gap: float, length: float) -> str:
    if z > gap / 2 + length or z < -gap / 2 - length:
        return "lost"
    if z > gap / 2:
        return "right tube"
    return "left tube" if z < -gap / 2 else "gap"


def fly(workdir: Path, cells: int = 12, gap: float = 20.0, depth: float = 3.0, volt: float = 10.0,
        temperature: float = 0.0, steps: int = 20_000, update: int = 25, seed: int = 9172345,
        name: str = "fly", threads: int = 4, tube: np.ndarray | None = None, flexible: bool = False) -> dict:
    """Le C60 part du tube de droite ; ses charges et celles des tubes sont recalculees toutes les `update` iterations.

    flexible : les tubes vibrent et se deforment, tenus par un anneau fige de deux mailles au milieu de leur corps
    (ils absorbent alors le choc du C60) ; sinon ils sont entierement figes."""
    errors = []
    for bound, v in (("cells", cells), ("gap_A", gap), ("depth_A", depth), ("voltage_V", volt),
                     ("temperature_K", temperature), ("steps", steps), ("update", update)):
        check(bound, v, errors)
    if errors:
        return {"ok": False, "errors": errors}
    from lammps import lammps as Lammps

    left, right, ball = setup(cells, gap, depth, tube)
    length = float(right[:, 2].max()) - gap / 2               # calotte comprise
    n_tubes, n_all = len(left) + len(right), len(left) + len(right) + 60
    zeros = [np.zeros(len(g)) for g in (left, right, ball)]
    nt.write_data(workdir / f"{name}.data", [left, right, ball], (-gap / 2 - length - 60.0, gap / 2 + length + 60.0), zeros)
    pot, every = nt.potential(), max(update, (steps // 200) // update * update)
    mid = gap / 2 + tube_length(cells) / 2                       # milieu du corps de chaque tube
    if flexible:
        hold = f"""region clampR block INF INF INF INF {mid - nt.AY:.4f} {mid + nt.AY:.4f} units box
region clampL block INF INF INF INF {-mid - nt.AY:.4f} {-mid + nt.AY:.4f} units box
region clamp union 2 clampL clampR
group clamp region clamp
group free subtract all clamp
neigh_modify one 6000 page 120000
fix hold clamp setforce 0.0 0.0 0.0"""
    else:
        hold = """group free union ball
# tubes figes : leurs interactions mutuelles ne servent a rien, on les retire de la liste de voisins
neigh_modify exclude group tubes tubes one 6000 page 120000
fix hold tubes setforce 0.0 0.0 0.0"""
    lmp = Lammps(cmdargs=["-screen", "none", "-log", (workdir / f"log.{name}.lammps").as_posix(),
                          "-sf", "omp", "-pk", "omp", str(threads)])
    text = f"""# genere par fullerene_shuttle - ne pas editer a la main
units metal
atom_style charge
atom_modify map yes
boundary f f f
read_data "{(workdir / f'{name}.data').as_posix()}"
pair_style hybrid/overlay airebo 3.0 1 0 coul/cut {COUL_CUT}
pair_coeff * * airebo "{pot}" C C C
pair_coeff * * coul/cut
group tubes type 1 2
group ball type 3
{hold}
# le C60 se relaxe SANS les charges, son centre de masse retenu par un ressort : sans lui, la minimisation
# l'aspire au fond du tube (le puits de van der Waals fait 3 eV) et il ne part plus de la position demandee
fix tether ball spring tether 500.0 0.0 0.0 {gap / 2 + depth:.6f} 0.0
min_style cg
minimize 1e-10 1e-10 2000 20000
unfix tether
reset_timestep 0
timestep {nt.DT_PS}
compute cm ball com
thermo_style custom step temp pe ke c_cm[3]
thermo {update}
fix move free nve
dump traj all custom {every} "{(workdir / f'{name}.lammpstrj').as_posix()}" id type q x y z
dump_modify traj sort id
"""
    (workdir / f"in.{name}.lammps").write_text(text)
    try:
        lmp.commands_string(text)
        if temperature > 0:
            lmp.command(f"velocity ball create {temperature} {seed} mom yes rot yes dist gaussian")

        def positions():
            return np.ctypeslib.as_array(lmp.gather_atoms("x", 1, 3)).reshape(-1, 3).copy()

        el = Electro(left, right, ball)
        rows, q_ball, intact = [], 0.0, True
        for seg in range(steps // update):
            x = positions()
            spread = np.linalg.norm(x[n_tubes:] - x[n_tubes:].mean(axis=0), axis=1)
            if spread.max() > 1.5 * c60_radius():                # le C60 s'est ouvert ou fragmente : on arrete la
                intact = False
                break
            q, link, phi = el.update(x[n_tubes:], volt, q_ball, tubes=x[:n_tubes] if flexible else None)
            q_ball = float(q[n_tubes:].sum())
            lmp.scatter_atoms("q", 1, 1, (ctypes.c_double * n_all)(*q))
            z_cm = float(x[n_tubes:, 2].mean())
            rows.append((seg * update * nt.DT_PS, z_cm, q_ball, link or "-", phi))
            if location(z_cm, gap, length) == "lost":
                break
            try:
                lmp.command(f"run {update}")
            except Exception as exc:                             # atomes sortis de la boite : le C60 a eclate
                if "Lost atoms" not in str(exc):
                    raise
                intact = False
                break
    finally:
        lmp.close()

    t, z, qb = (np.array([r[k] for r in rows]) for k in (0, 1, 2))
    links = [r[3] for r in rows]
    speed = np.gradient(z, t) * 100.0 if len(t) > 2 else np.zeros(len(t))        # A/ps -> m/s
    free = [k for k, l in enumerate(links) if l == "-"]
    first_free = free[0] if free else None
    arrive = next((k for k in range(first_free or 0, len(links)) if first_free is not None and links[k] == "L"), None)
    stride = max(1, len(rows) // 400)
    res = {"ok": True, "cells": cells, "gap_A": gap, "depth_A": depth, "capped": tube is not None, "flexible": flexible,
           "voltage_V": volt, "temperature_K": temperature,
           "steps": steps, "update": update, "seed": seed, "simulated_ps": round(float(t[-1]), 3),
           "time_ps": t[::stride].round(3).tolist(), "z_A": z[::stride].round(3).tolist(),
           "q_ball_e": qb[::stride].round(5).tolist(), "link": links[::stride],
           "phi_ball_V": [round(r[4], 3) for r in rows[::stride]],
           "release_ps": round(float(t[first_free]), 3) if first_free is not None else None,
           "arrival_ps": round(float(t[arrive]), 3) if arrive is not None else None,
           "q_flight_e": round(float(qb[first_free]), 5) if first_free is not None else None,
           "max_speed_m_s": round(float(np.abs(speed).max()), 1),
           "final_location": location(float(z[-1]), gap, length), "final_q_e": round(float(qb[-1]), 5),
           "c60_intact": intact}
    # energie cinetique du centre de masse au plus vite, a comparer au puits de van der Waals (environ 3 eV)
    res["max_kinetic_eV"] = round(0.5 * MASS_C60_KG * float(np.abs(speed).max()) ** 2 / 1.602177e-19, 2)
    if first_free is not None:                                   # la charge d'un conducteur isole doit se conserver
        run_end = next((k for k in range(first_free, len(links)) if links[k] != "-"), len(links))
        res["charge_drift_e"] = float(np.abs(qb[first_free:run_end] - qb[first_free]).max())
        res["flight_ps"] = round(float(t[run_end - 1] - t[first_free]), 3)
    res["crossed"] = arrive is not None
    return res


# --------------------------------------------------------------------------- #
# Plausibilite
# --------------------------------------------------------------------------- #
def plausibility(res: dict) -> list:
    warn = []

    def rng(key, value, label):
        lo, hi = PLAUSIBLE[key]
        if value is not None and not (lo <= value <= hi):
            warn.append(f"{label} = {value} hors de la plage plausible [{lo}, {hi}]")

    w = res.get("well", {})
    rng("wall_gap_A", w.get("wall_gap_A"), "distance C60-paroi (A)")
    rng("binding_eV", w.get("binding_eV"), "energie d'encapsulation (eV)")
    rng("retention_nN", w.get("retention_max_nN"), "force de retenue (nN)")
    for key, run in res.items():
        if key.startswith("fly") and isinstance(run, dict):
            rng("charge_drift_e", run.get("charge_drift_e"), f"derive de charge en vol, {key} (e)")
    return warn


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Navette a fullerene entre deux nanotubes (10,10)")
    ap.add_argument("task", choices=["well", "threshold", "fly", "all"])
    ap.add_argument("--workdir", default="runs_fs")
    ap.add_argument("--cells", type=int, default=12)
    ap.add_argument("--gap", type=float, default=20.0)
    ap.add_argument("--depth", type=float, default=3.0)
    ap.add_argument("--volt", type=float, default=None, help="tension pour 'fly' (defaut : 1,3 x le seuil statique)")
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--update", type=int, default=25)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=9172345)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--capped", action="store_true", help="tubes fermes au bout arriere par un demi-C240")
    ap.add_argument("--label", default="", help="suffixe de la cle du vol, pour ne pas ecraser un vol precedent")
    ap.add_argument("--flexible", action="store_true", help="tubes flexibles, tenus par un anneau au milieu (vol seulement)")
    a = ap.parse_args()
    wd = Path(a.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    out_file = wd / "results.json"
    res = json.loads(out_file.read_text()) if out_file.exists() else {}

    def save():
        res["warnings"] = plausibility(res)
        out_file.write_text(json.dumps(res, indent=1))

    def brief(d):
        return {k: v for k, v in d.items() if not isinstance(v, list)}

    tube = relax_cap(wd, a.cells) if a.capped else None
    tag = "_capped" if a.capped else ""
    if a.task in ("well", "all"):
        res["well"] = well(wd, a.cells)
        save()
        print("puits :", brief(res["well"]))
    if a.task in ("threshold", "all"):
        res["threshold" + tag] = threshold(a.cells, a.gap, a.depth, res.get("well"), tube=tube)
        save()
        print("seuil :", brief(res["threshold" + tag]))
    if a.task in ("fly", "all"):
        thr = res.get("threshold" + tag, {}).get("threshold_V")
        volt = a.volt or (round(1.3 * thr, 1) if thr else 10.0)
        key = f"fly{tag}{'_flex' if a.flexible else ''}_{volt:g}V" + (f"_{a.label}" if a.label else "")
        res[key] = fly(wd, a.cells, a.gap, a.depth, volt, a.temp, a.steps, a.update, a.seed, name=key, threads=a.threads, tube=tube,
                       flexible=a.flexible)
        if not res[key].get("ok"):
            sys.exit(" ; ".join(res[key]["errors"]))
        save()
        print("vol :", brief(res[key]))
    save()
    print("avertissements :", res["warnings"] or "aucun")
