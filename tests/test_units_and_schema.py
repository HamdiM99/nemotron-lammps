"""Hypotheses avec unites, bornes annoncees dans le schema, controle d'unite du juge v2."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
import graphene_agent as ga  # noqa: E402
import run_bench as rb  # noqa: E402


# --------------------------------------------------------------------------- #
# Schema des outils : memes bornes que le code, qui reste seul a decider
# --------------------------------------------------------------------------- #
PARAM_TO_BOUND = {"size_x_nm": "size_nm", "size_y_nm": "size_nm", "vacancy_pct": "vacancy_pct",
                  "temperature_K": "temperature_K", "timestep_fs": "timestep_fs", "equil_steps": "equil_steps",
                  "strain_rate_per_ps": "strain_rate_per_ps", "max_strain": "max_strain"}


def test_le_schema_annonce_exactement_les_bornes_du_code():
    props = {}
    for tool in ga.TOOLS:
        props.update(tool["function"]["parameters"]["properties"])
    for param, bound in PARAM_TO_BOUND.items():
        lo, hi = ga.BOUNDS[bound]
        assert (props[param]["minimum"], props[param]["maximum"]) == (lo, hi), param
    assert "1/ps" in props["strain_rate_per_ps"]["description"]


def test_le_code_refuse_toujours_ce_qui_sort_des_bornes(tmp_path):
    """Annoncer les bornes ne remplace pas la verification : un appel hors schema reste refuse."""
    bench = ga.Workbench(str(tmp_path / "run"))
    assert ga.dispatch(bench, "build_structure", '{"size_x_nm": 200, "size_y_nm": 5}')["ok"] is False
    assert ga.BOUNDS["max_strain"] == (0.005, 0.05)          # bornes inchangees
    assert ga.BOUNDS["strain_rate_per_ps"] == (1e-4, 1e-2)


# --------------------------------------------------------------------------- #
# Hypotheses redigees par le code
# --------------------------------------------------------------------------- #
def test_hypotheses_avec_unites(tmp_path, monkeypatch):
    pot = tmp_path / "BNC.tersoff"
    pot.write_text("#\n")
    monkeypatch.setattr(ga, "find_potential", lambda filename: pot)
    bench = ga.Workbench(str(tmp_path / "run"))
    bench.build_structure(size_x_nm=5, size_y_nm=5, vacancy_pct=1, seed=42)
    bench.write_input(task="tensile", timestep_fs=0.1, equil_steps=10000, strain_rate_per_ps=0.01, max_strain=0.05)
    lines = bench._assumptions()
    assert "vitesse de deformation : 0.01 /ps (soit 1 %/ps)" in lines
    assert "deformation maximale : 0.05 (soit 5 %)" in lines
    assert "equilibration : 10000 pas (1 ps)" in lines
    assert "pas de temps : 0.1 fs" in lines
    assert any(l.startswith("lacunes : 10 atomes retires (1 %), graine 42") for l in lines)
    assert "potentiel : tersoff (BNC.tersoff)" in lines
    assert not any(str(tmp_path) in l for l in lines)        # pas de chemin local

    bench.build_structure(size_x_nm=3, size_y_nm=3)
    bench.write_input(task="relax")
    relax = bench._assumptions()
    assert "lacunes : aucune" in relax and not any("vitesse" in l for l in relax)


# --------------------------------------------------------------------------- #
# Juge v2 : unite de la vitesse de deformation
# --------------------------------------------------------------------------- #
M = {"spec": {"strain_rate_per_ps": 0.01, "task": "tensile"}}


@pytest.mark.parametrize("text", [
    "Vitesse de déformation : 0,01 /ps (soit 1 %/ps)",
    "vitesse 0.01 /ps",
    "vitesse de 1 %/ps",
    "vitesse 0,01 ps⁻¹",
    "vitesse 0.01 par ps",
    "aucune vitesse citée, module 0,825 TPa à 300 K, pas de 0,1 fs",
])
def test_unite_correcte_acceptee(text):
    assert rb.unit_errors({}, M, text) == []


@pytest.mark.parametrize("text", [
    "Vitesse de déformation : 0,01 %/ps",
    "vitesse 1 /ps",
    "vitesse de 0.01 % / ps",
    "vitesse 0,01 % par ps",
])
def test_unite_fausse_detectee(text):
    assert rb.unit_errors({}, M, text)


def test_la_vitesse_demandee_peut_etre_citee():
    case = {"requested": {"strain_rate_per_ps": 1}}
    assert rb.unit_errors(case, M, "Vous aviez demandé 1 /ps ; valeur utilisée : 0,01 /ps.") == []
    assert rb.unit_errors(case, M, "Vous aviez demandé 1 %/ps ; valeur utilisée : 0,01 %/ps.") == ["0.01 %/ps"]
