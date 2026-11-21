"""Tests de nanotube_lab : geometrie, solveur de charges, bornes. Rapides, sans LAMMPS (sauf le test slow)."""
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nanotube_lab as nt  # noqa: E402


# --------------------------------------------------------------------------- #
# Geometrie
# --------------------------------------------------------------------------- #
def test_tube_armchair_nombre_d_atomes_et_rayon():
    tube = nt.armchair_tube(5, 20)
    assert tube.shape == (400, 3)                                   # 4n atomes par maille
    r = np.hypot(tube[:, 0], tube[:, 1])
    assert np.allclose(r, nt.radius(5)) and nt.radius(5) == pytest.approx(3.39, abs=0.01)


def test_chaque_atome_a_trois_voisins_sauf_aux_bouts():
    tube = nt.armchair_tube(5, 12)
    d = np.linalg.norm(tube[:, None] - tube[None, :], axis=-1)
    coord = ((d > 0.1) & (d < 1.6)).sum(axis=1)
    assert set(coord.tolist()) <= {2, 3}
    assert (coord == 2).sum() == 20                                 # 2n atomes sous-coordonnes par bout
    bonds = d[(d > 0.1) & (d < 1.6)]
    assert bonds.min() > 1.38 and bonds.max() < 1.43                # liaisons proches de 1,42 A apres enroulement


def test_ecart_entre_parois():
    inner, outer = nt.shuttle(5, 10)
    assert nt.radius(10) - nt.radius(5) == pytest.approx(3.39, abs=0.01)
    lo, hi = nt.PLAUSIBLE["interlayer_spacing_A"]
    assert lo <= nt.radius(10) - nt.radius(5) <= hi
    assert len(outer) == 2 * len(inner)


def test_tete_beche_symetrique_et_ecart_respecte():
    il, ol, ir, orr = nt.head_to_head(5, 10, gap=15.0)
    assert ol[:, 2].max() == pytest.approx(-7.5) and orr[:, 2].min() == pytest.approx(7.5)
    il2, _, ir2, _ = nt.head_to_head(5, 10, gap=15.0, protrusion=2.0)
    assert il2[:, 2].max() - il[:, 2].max() == pytest.approx(2.0)   # le coeur gauche avance vers +z
    assert ir[:, 2].min() - ir2[:, 2].min() == pytest.approx(2.0)   # le coeur droit avance vers -z


# --------------------------------------------------------------------------- #
# Electrostatique
# --------------------------------------------------------------------------- #
def test_matrice_symetrique_et_terme_propre():
    pos = nt.armchair_tube(5, 4)
    a = nt.interaction_matrix(pos)
    assert np.allclose(a, a.T)
    assert a[0, 0] == pytest.approx(nt.K_COULOMB / (math.sqrt(math.pi) * nt.R_GAUSS))
    far = np.linalg.norm(pos[0] - pos[-1])
    assert a[0, -1] == pytest.approx(nt.K_COULOMB / far, rel=1e-6)  # loin : coulombien pur


def test_charges_lineaires_en_tension_et_potentiel_impose():
    pos = nt.armchair_tube(5, 8)
    q1 = nt.solve_charges(pos, np.full(len(pos), 1.0))
    q6 = nt.solve_charges(pos, np.full(len(pos), 6.0))
    assert np.allclose(q6, 6 * q1)
    assert np.allclose(nt.interaction_matrix(pos) @ q6, 6.0)        # chaque atome est bien a 6 V
    assert q6.sum() > 0


def test_profil_de_charge_plus_fort_aux_bouts_et_dans_la_plage_de_l_article():
    c = nt.charges(5, 20, 6.0)
    cell = c["q_cell_e"]
    assert cell[0] > 2 * cell[len(cell) // 2] and cell[-1] > 2 * cell[len(cell) // 2]
    assert cell[0] == pytest.approx(cell[-1], rel=0.05)             # tube symetrique
    # ICRA 2010, Fig. 2(a) : 5 a 34 x 10^-3 e pour un tube de 5 nm a 6 V
    assert 5e-3 <= c["q_cell_min_e"] and c["q_cell_max_e"] <= 34e-3
    assert c["capacitance_aF"] == pytest.approx(c["q_total_e"] / 6.0 * nt.E_TO_AF_PER_V, rel=1e-3)


def test_deux_navettes_charges_opposees_et_attraction():
    groups = nt.head_to_head(5, 8, gap=10.0)
    pos = np.vstack(groups)
    half = len(groups[0]) + len(groups[1])
    q = nt.solve_charges(pos, np.where(np.arange(len(pos)) < half, 0.5, -0.5))
    assert q[:half].sum() == pytest.approx(-q[half:].sum(), rel=1e-6)   # antisymetrie
    sel = np.zeros(len(pos), dtype=bool)
    sel[:len(groups[0])] = True
    assert nt.axial_force(pos, q, sel) > 0                          # le coeur gauche est tire vers +z


def test_le_seuil_augmente_avec_l_ecart():
    f = nt.force(5, 8, gaps=(10.0, 20.0), retraction_nN=0.87)
    v10, v20 = (g["threshold_V"] for g in f["gaps"])
    assert v20 > v10 > 0
    for g in f["gaps"]:                                             # la force croit quand les coeurs approchent
        forces = [p["force_nN_per_V2"] for p in g["scan"]]
        assert forces == sorted(forces)


# --------------------------------------------------------------------------- #
# Bornes et plausibilite
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kwargs", [{"gap": 500.0}, {"volt": 100.0}, {"temperature": 5000.0}, {"steps": 10}])
def test_fermeture_refusee_hors_bornes(tmp_path, kwargs):
    out = nt.close_shuttles(tmp_path, **kwargs)
    assert out["ok"] is False and "hors bornes" in out["errors"][0]
    assert not list(tmp_path.iterdir())                             # rien n'a ete ecrit ni lance


def test_oscillateur_refuse_hors_bornes(tmp_path):
    assert nt.oscillator(tmp_path, steps=10_000_000)["ok"] is False


def test_plausibilite_signale_une_valeur_aberrante():
    ok = {"static": {"spacing_A": 3.39, "adhesion_J_m2": 0.27}, "oscillator": {"freq_ratio": 0.995}}
    assert nt.plausibility(ok) == []
    bad = {"static": {"spacing_A": 3.39, "adhesion_J_m2": 5.0}, "oscillator": {"freq_ratio": 4.0}}
    assert len(nt.plausibility(bad)) == 2


# --------------------------------------------------------------------------- #
# Integration LAMMPS
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_integration_statique(tmp_path):
    if not shutil.which("lmp") or nt.ga.find_potential("CH.airebo") is None:
        pytest.skip("LAMMPS ou CH.airebo introuvable")
    s = nt.static(tmp_path, 5, 20)
    assert 0.5 < s["retraction_force_nN"] < 1.5
    lo, hi = nt.PLAUSIBLE["adhesion_J_m2"]
    assert lo <= s["adhesion_J_m2"] <= hi
    assert 0 < s["slide_barrier_meV_per_atom"] < 1 and 0 < s["rotate_barrier_meV_per_atom"] < 1
    assert s["rotate"]["meV_per_atom"][0] == pytest.approx(s["rotate"]["meV_per_atom"][18], abs=1e-3)   # periode 18 degres
