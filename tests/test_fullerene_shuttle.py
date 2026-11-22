"""Tests de fullerene_shuttle : geometrie, modele de charge, bornes. Rapides, sans LAMMPS (sauf les tests slow)."""
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fullerene_shuttle as fs  # noqa: E402
import nanotube_lab as nt  # noqa: E402


# --------------------------------------------------------------------------- #
# Geometrie
# --------------------------------------------------------------------------- #
def test_c60_soixante_atomes_trois_voisins_quatre_vingt_dix_liaisons():
    ball = fs.c60()
    assert ball.shape == (60, 3)
    d = np.linalg.norm(ball[:, None] - ball[None, :], axis=-1)
    bonded = (d > 0.1) & (d < 1.6)
    assert set(bonded.sum(axis=1).tolist()) == {3}
    assert bonded.sum() // 2 == 90
    assert np.allclose(d[bonded], fs.C60_BOND)
    assert np.allclose(np.linalg.norm(ball, axis=1), fs.c60_radius())
    assert fs.c60_radius() == pytest.approx(3.55, abs=0.03)        # rayon experimental du C60


def test_le_c60_loge_dans_le_tube_10_10():
    lo, hi = fs.PLAUSIBLE["wall_gap_A"]
    assert lo <= nt.radius(fs.TUBE_N) - fs.c60_radius() <= hi


def test_montage_bouches_et_position_du_c60():
    left, right, ball = fs.setup(cells=12, gap=20.0, depth=3.0)
    assert left[:, 2].max() == pytest.approx(-10.0) and right[:, 2].min() == pytest.approx(10.0)
    assert ball[:, 2].mean() == pytest.approx(13.0)
    assert np.allclose(ball[:, :2].mean(axis=0), 0.0, atol=1e-9)   # sur l'axe
    assert len(left) == len(right) == 40 * 12


# --------------------------------------------------------------------------- #
# Calotte
# --------------------------------------------------------------------------- #
def test_c240_trois_voisins_et_anneau_equatorial_armchair():
    ball = fs.c240_unit()
    assert ball.shape == (240, 3) and len(np.unique(ball.round(6), axis=0)) == 240
    nb = fs.neighbours(ball)
    assert all(i in nb[j] for i in range(240) for j in nb[i])       # voisinage reciproque : graphe 3-regulier
    belt = set(np.where(np.abs(ball[:, 2]) < 0.05)[0].tolist())
    assert len(belt) == 20 and (ball[:, 2] > 0.05).sum() == 110
    for i in belt:                                                  # comme un anneau armchair : 1 voisin dans l'anneau, 1 dessus, 1 dessous
        z = sorted(round(float(ball[j, 2]), 1) for j in nb[i] if j not in belt)
        assert sum(j in belt for j in nb[i]) == 1 and z[0] < 0 < z[1]


def test_tube_a_calotte_ferme_d_un_seul_cote():
    tube = fs.capped_tube(cells=8)
    assert len(tube) == 40 * 8 + 110
    q = fs.cap_quality(tube)
    assert q["other"] == 0 and q["two_neighbours"] == 20            # seuls les 20 atomes de la bouche ont 2 voisins
    assert 1.35 < q["bond_min_A"] and q["bond_max_A"] < 1.50        # deja correct avant la relaxation AIREBO
    assert np.allclose(tube[:320], nt.armchair_tube(fs.TUBE_N, 8, 0.0))    # le corps du tube n'est pas touche
    # avant la relaxation AIREBO la calotte est un peu renflee (7,1 A contre 6,78 A ; 6,9 A une fois relaxee)
    assert np.hypot(tube[320:, 0], tube[320:, 1]).max() < nt.radius(fs.TUBE_N) + 0.5


def test_montage_avec_calottes_symetrique():
    tube = fs.capped_tube(cells=8)
    left, right, _ = fs.setup(8, 20.0, 0.0, tube)
    assert right[:, 2].min() == pytest.approx(10.0) and left[:, 2].max() == pytest.approx(-10.0)
    assert np.allclose(np.sort(left[:, 2]), np.sort(-right[:, 2]))


# --------------------------------------------------------------------------- #
# Contact electrique
# --------------------------------------------------------------------------- #
def test_contact_dans_le_tube_isole_au_milieu_de_l_ecart():
    left, right, ball = fs.setup(12, 20.0, 3.0)
    assert fs.link_of(ball, left, right) == "R"
    assert fs.link_of(ball - [0, 0, 13.0], left, right) is None    # centre au milieu de l'ecart
    assert fs.link_of(ball - [0, 0, 26.0], left, right) == "L"     # position symetrique, dans le tube de gauche


def test_emplacement_du_c60():
    length = fs.tube_length(12)
    assert fs.location(13.0, 20.0, length) == "right tube"
    assert fs.location(0.0, 20.0, length) == "gap"
    assert fs.location(-20.0, 20.0, length) == "left tube"
    assert fs.location(-10.0 - length - 1.0, 20.0, length) == "lost"


# --------------------------------------------------------------------------- #
# Modele de charge
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def small():
    left, right, ball = fs.setup(cells=8, gap=20.0, depth=3.0)
    return left, right, ball, fs.Electro(left, right, ball)


def test_le_noyau_est_celui_de_nanotube_lab():
    pos = np.vstack([nt.armchair_tube(5, 3), fs.c60((0, 0, 20.0))])
    assert np.allclose(fs.block(pos, pos), nt.interaction_matrix(pos))


def test_c60_relie_il_prend_le_potentiel_et_le_signe_du_tube(small):
    left, right, ball, el = small
    q, link, phi = el.update(ball, 10.0, 0.0)
    assert link == "R" and phi == pytest.approx(-5.0)
    assert np.allclose(el.a @ q, np.r_[np.full(len(left), 5.0), np.full(len(right) + 60, -5.0)])
    q_ball = q[el.nt_:].sum()
    assert q_ball < 0                                              # tube de droite negatif, le C60 aussi
    assert fs.ball_force_z(el.tubes, ball, q) < 0                  # tire vers -z, c'est-a-dire vers le tube de gauche


def test_c60_isole_charge_conservee_et_equipotentiel(small):
    left, right, ball, el = small
    mid = ball - [0, 0, 13.0]
    q, link, phi = el.update(mid, 10.0, -0.25)
    assert link is None
    assert q[el.nt_:].sum() == pytest.approx(-0.25, abs=1e-9)      # la charge totale imposee est respectee
    pot = el.a @ q
    assert np.allclose(pot[el.nt_:], phi)                          # les 60 atomes sont au meme potentiel
    assert np.allclose(pot[:len(left)], 5.0) and np.allclose(pot[len(left):el.nt_], -5.0)


def test_c60_neutre_au_milieu_reste_neutre_et_a_potentiel_nul(small):
    left, right, ball, el = small
    q, _, phi = el.update(ball - [0, 0, 13.0], 10.0, 0.0)
    assert q[el.nt_:].sum() == pytest.approx(0.0, abs=1e-9)
    assert phi == pytest.approx(0.0, abs=0.05)                     # montage symetrique : potentiel median
    assert abs(fs.ball_force_z(el.tubes, ball - [0, 0, 13.0], q)) < 1e-3


def test_les_charges_sont_lineaires_en_tension(small):
    left, right, ball, el = small
    q1, _, _ = el.update(ball, 1.0, 0.0)
    q7, _, _ = el.update(ball, 7.0, 0.0)
    assert np.allclose(q7, 7 * q1)


def test_balayage_du_seuil_charge_gardee_apres_la_separation():
    out = fs.threshold(cells=8, gap=20.0, depth=3.0)
    assert out["ok"] and out["scan"][0]["link"] == "R"
    free = [r for r in out["scan"] if r["link"] is None]
    assert free and out["release_at_A"] == free[0]["d_A"]
    assert len({r["q_ball_e_per_V"] for r in free}) == 1           # charge constante en vol libre
    assert out["q_ball_flight_e_per_V"] < 0
    assert all(r["pull_nN_per_V2"] > 0 for r in free)              # en vol, le C60 est tire vers le tube d'en face


# --------------------------------------------------------------------------- #
# Bornes et plausibilite
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kwargs", [{"gap": 10.0}, {"gap": 80.0}, {"volt": 100.0}, {"steps": 10}, {"update": 1},
                                    {"cells": 100}, {"depth": 50.0}])
def test_vol_refuse_hors_bornes(tmp_path, kwargs):
    out = fs.fly(tmp_path, **kwargs)
    assert out["ok"] is False and "hors bornes" in out["errors"][0]
    assert not list(tmp_path.iterdir())                            # rien n'a ete ecrit ni lance


def test_seuil_refuse_hors_bornes():
    assert fs.threshold(gap=5.0)["ok"] is False


def test_plausibilite_signale_une_valeur_aberrante():
    ok = {"well": {"wall_gap_A": 3.24, "binding_eV": 3.2, "retention_max_nN": 0.6}, "fly_10V": {"charge_drift_e": 0.0}}
    assert fs.plausibility(ok) == []
    bad = {"well": {"wall_gap_A": 3.24, "binding_eV": 30.0, "retention_max_nN": 0.6}, "fly_10V": {"charge_drift_e": 0.01}}
    assert len(fs.plausibility(bad)) == 2


# --------------------------------------------------------------------------- #
# Integration LAMMPS
# --------------------------------------------------------------------------- #
def _lammps_ok():
    return shutil.which("lmp") and nt.ga.find_potential("CH.airebo") is not None


@pytest.mark.slow
def test_integration_puits(tmp_path):
    if not _lammps_ok():
        pytest.skip("LAMMPS ou CH.airebo introuvable")
    w = fs.well(tmp_path, cells=8)
    lo, hi = fs.PLAUSIBLE["binding_eV"]
    assert lo <= w["binding_eV"] <= hi
    assert w["retention_max_nN"] > 0 and abs(w["retention_max_at_A"]) < 6      # la retenue est maximale pres de la bouche
    assert abs(w["energy_eV"][0] - w["energy_eV"][2]) < 0.05                   # fond de puits plat au milieu du tube


@pytest.mark.slow
def test_integration_vol_court(tmp_path):
    if not _lammps_ok():
        pytest.skip("LAMMPS ou CH.airebo introuvable")
    pytest.importorskip("lammps")
    out = fs.fly(tmp_path, cells=8, gap=20.0, depth=3.0, volt=5.0, steps=1000, update=50, threads=2)
    assert out["ok"] and out["link"][0] == "R"
    assert out["z_A"][0] == pytest.approx(13.0, abs=0.1)           # la minimisation n'a pas aspire le C60 au fond du tube
    assert out["q_ball_e"][0] < 0                                  # a la bouche, il porte le signe de son tube
    assert out["z_A"][-1] > out["z_A"][0]                          # 5 V, sous le seuil : le tube le reprend
    assert (tmp_path / "fly.lammpstrj").exists()
