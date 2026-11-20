"""Tests de graphene_agent : garde-fous deterministes, lecture du log, integration LAMMPS.

  pytest                 # tests rapides, sans LAMMPS ni LLM
  pytest -m slow         # integration LAMMPS reelle (~1 min)
"""
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import graphene_agent as ga  # noqa: E402


@pytest.fixture
def bench(tmp_path):
    return ga.Workbench(str(tmp_path / "run"))


@pytest.fixture
def fake_potential(monkeypatch, tmp_path):
    """Rend write_input independant de l'installation LAMMPS."""
    pot = tmp_path / "BNC.tersoff"
    pot.write_text("# faux potentiel pour les tests\n")
    monkeypatch.setattr(ga, "find_potential", lambda filename: pot)
    return pot


@pytest.fixture
def ready(bench, fake_potential):
    """Un banc avec structure et input valides, aucun run lance."""
    assert bench.build_structure(size_x_nm=2, size_y_nm=2)["ok"]
    assert bench.write_input(task="tensile")["ok"]
    return bench


# --------------------------------------------------------------------------- #
# Bornes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kwargs, fragment", [
    ({"size_x_nm": 200, "size_y_nm": 200}, "size_nm=200"),
    ({"size_x_nm": 0.5, "size_y_nm": 5}, "size_nm=0.5"),
    ({"size_x_nm": 5, "size_y_nm": 5, "vacancy_pct": 50}, "vacancy_pct=50"),
    ({"size_x_nm": 5, "size_y_nm": 5, "vacancy_pct": -1}, "vacancy_pct=-1"),
    ({"size_x_nm": "5", "size_y_nm": 5}, "size_nm=5"),
])
def test_build_structure_refuse_hors_bornes(bench, kwargs, fragment):
    out = bench.build_structure(**kwargs)
    assert out["ok"] is False
    assert any(fragment in e and "hors bornes" in e for e in out["errors"])
    assert bench.structure is None
    assert not (bench.dir / "graphene.data").exists()


def test_build_structure_refuse_budget_atomes(bench, monkeypatch):
    monkeypatch.setattr(ga, "MAX_ATOMS", 100)
    out = bench.build_structure(size_x_nm=5, size_y_nm=5)
    assert out["ok"] is False and "budget" in out["errors"][0]


@pytest.mark.parametrize("kwargs, fragment", [
    ({"timestep_fs": 5}, "timestep_fs=5"),
    ({"timestep_fs": 0.01}, "timestep_fs=0.01"),
    ({"temperature_K": 0}, "temperature_K=0"),
    ({"temperature_K": 5000}, "temperature_K=5000"),
    ({"equil_steps": 10}, "equil_steps=10"),
    ({"strain_rate_per_ps": 1.0}, "strain_rate_per_ps=1.0"),
    ({"max_strain": 0.2}, "max_strain=0.2"),
])
def test_write_input_refuse_hors_bornes(bench, fake_potential, kwargs, fragment):
    assert bench.build_structure(size_x_nm=2, size_y_nm=2)["ok"]
    out = bench.write_input(task="tensile", **kwargs)
    assert out["ok"] is False
    assert any(fragment in e and "hors bornes" in e for e in out["errors"])
    assert bench.spec is None
    assert not (bench.dir / "in.full.lammps").exists()


def test_write_input_refuse_sans_structure(bench, fake_potential):
    out = bench.write_input(task="tensile")
    assert out["ok"] is False and "build_structure" in out["errors"][0]


def test_write_input_refuse_tache_et_potentiel_inconnus(bench, fake_potential):
    bench.build_structure(size_x_nm=2, size_y_nm=2)
    out = bench.write_input(task="bandgap", potential="dft")
    assert out["ok"] is False
    assert any("task" in e for e in out["errors"])
    assert any("potential" in e for e in out["errors"])


def test_write_input_avertit_au_dela_de_1_fs(bench, fake_potential):
    bench.build_structure(size_x_nm=2, size_y_nm=2)
    out = bench.write_input(task="tensile", timestep_fs=2.0)
    assert out["ok"] is True and out["warnings"]


def test_input_genere_contient_le_dump_de_trajectoire(ready):
    txt = (ready.dir / "in.full.lammps").read_text()
    assert "dump traj all custom" in txt and "traj.full.lammpstrj" in txt


def test_structure_reproductible_avec_la_graine(tmp_path):
    a, b = ga.Workbench(str(tmp_path / "a")), ga.Workbench(str(tmp_path / "b"))
    ra = a.build_structure(size_x_nm=3, size_y_nm=3, vacancy_pct=2, seed=7)
    rb = b.build_structure(size_x_nm=3, size_y_nm=3, vacancy_pct=2, seed=7)
    assert ra == rb and ra["n_vacancies"] > 0
    assert (a.dir / "graphene.data").read_text() == (b.dir / "graphene.data").read_text()


# --------------------------------------------------------------------------- #
# Smoke obligatoire avant full
# --------------------------------------------------------------------------- #
def test_run_refuse_sans_input(bench):
    out = bench.run(mode="smoke")
    assert out["ok"] is False and "write_input" in out["errors"][0]


def test_full_refuse_avant_smoke(ready):
    out = ready.run(mode="full")
    assert out["ok"] is False
    assert "smoke" in out["errors"][0]
    assert not (ready.dir / "log.full.lammps").exists()


def test_full_refuse_apres_smoke_echoue(ready, monkeypatch):
    monkeypatch.setattr(ga.shutil, "which", lambda name: None)   # LAMMPS introuvable
    assert ready.run(mode="smoke")["ok"] is False
    assert ready.smoke_ok is False
    assert ready.run(mode="full")["ok"] is False


def test_nouvel_input_ou_structure_invalide_le_smoke(ready):
    ready.smoke_ok = True
    ready.write_input(task="tensile", temperature_K=500)
    assert ready.smoke_ok is False
    ready.smoke_ok = True
    ready.build_structure(size_x_nm=2, size_y_nm=2)
    assert ready.smoke_ok is False and ready.spec is None


def test_mode_inconnu_refuse(ready):
    assert ready.run(mode="production")["ok"] is False


def test_parse_log_refuse_sans_run_complet(ready):
    out = ready.parse_log()
    assert out["ok"] is False and "run complet" in out["errors"][0]


# --------------------------------------------------------------------------- #
# dispatch : seuls les 4 outils declares
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["shell", "write_file", "_render", "__init__", "dir"])
def test_outil_inconnu_refuse(bench, name):
    out = ga.dispatch(bench, name, "{}")
    assert out == {"ok": False, "errors": [f"outil inconnu : {name}"]}


def test_arguments_invalides_refuses(bench):
    assert ga.dispatch(bench, "build_structure", "{pas du json")["ok"] is False
    assert ga.dispatch(bench, "build_structure", '{"size_x_nm": 5}')["ok"] is False
    out = ga.dispatch(bench, "run", '{"mode": "smoke", "raw_input": "shell rm -rf"}')
    assert out["ok"] is False and "arguments invalides" in out["errors"][0]


def test_les_outils_exposes_sont_exactement_quatre():
    assert [t["function"]["name"] for t in ga.TOOLS] == \
        ["build_structure", "write_input", "run", "parse_log"]


# --------------------------------------------------------------------------- #
# parse_thermo / linfit
# --------------------------------------------------------------------------- #
SAMPLE_LOG = """LAMMPS (22 Jul 2025 - Update 4)
Reading data file ...
  960 atoms
WARNING: une ligne d'avertissement au milieu
Per MPI rank memory allocation (min/avg/max) = 4.5 | 4.5 | 4.5 Mbytes
   Step          Temp          PotEng         KinEng         TotEng          Pxx            Pyy             Lx             Ly
         0   0             -7650.1         0              -7650.1         1200.5         1100.2         51.12          49.19
        50   0             -7658.6         0              -7658.6         0.01          -0.02           51.80          49.84
Loop time of 0.5 on 1 procs for 50 steps with 960 atoms

Minimization stats:
  Stopping criterion = energy tolerance
   Step          Temp          PotEng         KinEng         TotEng          Pxx            Pyy             Lx             Ly
         0   300           -7658.6         37.2           -7621.4         10.0           5.0            51.80          49.84
WARNING: Step 20 pas une ligne de donnees
        50   295.5         -7640.2         36.6           -7603.6        -5000.0         3.0            51.93          49.83
       100   301.2         -7622.9         37.3           -7585.6        -10000.0       -2.0            52.06          49.82
Loop time of 1.2 on 1 procs for 100 steps with 960 atoms
Total wall time: 0:00:02
"""


def test_parse_thermo_sur_un_log_exemple():
    blocks = ga.parse_thermo(SAMPLE_LOG)
    assert [len(b) for b in blocks] == [2, 3]
    assert blocks[0][-1]["PotEng"] == -7658.6
    assert blocks[0][-1]["Ly"] == 49.84
    assert blocks[1][0]["Temp"] == 300.0
    assert [r["Step"] for r in blocks[1]] == [0.0, 50.0, 100.0]
    assert set(blocks[1][0]) == {"Step", "Temp", "PotEng", "KinEng", "TotEng", "Pxx", "Pyy", "Lx", "Ly"}


def test_parse_thermo_garde_un_run_interrompu():
    tronque = SAMPLE_LOG.split("Loop time of 1.2")[0]
    blocks = ga.parse_thermo(tronque)
    assert [len(b) for b in blocks] == [2, 3]


def test_parse_thermo_log_vide():
    assert ga.parse_thermo("") == []
    assert ga.parse_thermo("ERROR: Unknown pair style\n") == []


def test_linfit():
    slope, r2 = ga.linfit([0, 1, 2, 3], [1, 3, 5, 7])
    assert slope == pytest.approx(2.0) and r2 == pytest.approx(1.0)


def test_sanity_detecte_nan_et_energie_aberrante(ready):
    n = ready.structure["n_atoms"]
    row = {"Step": 0, "Temp": 300.0, "PotEng": -7.9 * n}
    assert ready._sanity([[row]], "smoke") == []
    assert "diverge" in ready._sanity([[dict(row, PotEng=float("nan"))]], "smoke")[0]
    assert "hors plage" in ready._sanity([[dict(row, PotEng=-2.0 * n)]], "smoke")[0]
    assert ready._sanity([], "smoke") == ["aucune sortie thermo dans le log"]


# --------------------------------------------------------------------------- #
# Boucle agent avec un faux modele scripte (aucun appel reseau)
# --------------------------------------------------------------------------- #
class FakeClient:
    """Rejoue une liste de reponses : soit [(outil, args), ...], soit un texte final."""

    def __init__(self, script):
        self.script = list(script)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        step = self.script.pop(0)
        if isinstance(step, str):
            msg = SimpleNamespace(content=step, tool_calls=None)
        else:
            msg = SimpleNamespace(content="", tool_calls=[
                SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name=n, arguments=json.dumps(a)))
                for i, (n, a) in enumerate(step)])
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def read_trace(workdir):
    return [json.loads(l) for l in (Path(workdir) / "trace.jsonl").read_text(encoding="utf-8").splitlines()]


def test_agent_les_refus_reviennent_au_modele_et_sont_traces(tmp_path, fake_potential):
    client = FakeClient([
        [("build_structure", {"size_x_nm": 200, "size_y_nm": 200})],     # hors bornes
        [("build_structure", {"size_x_nm": 2, "size_y_nm": 2})],
        [("write_input", {"task": "tensile", "timestep_fs": 5})],        # hors bornes
        [("write_input", {"task": "tensile"})],
        [("run", {"mode": "full"})],                                     # full avant smoke
        [("exec", {"cmd": "lmp -in mon_input"})],                        # outil inconnu
        "Je m'arrete la.",
    ])
    wd = str(tmp_path / "agent")
    assert ga.run_agent("test", wd, client=client, model="faux") == "Je m'arrete la."
    trace = read_trace(wd)
    assert trace[0]["type"] == "start" and trace[-1]["type"] == "final"
    oks = [(r["name"], r["result"]["ok"]) for r in trace if r["type"] == "tool"]
    assert oks == [("build_structure", False), ("build_structure", True), ("write_input", False),
                   ("write_input", True), ("run", False), ("exec", False)]
    assert not (Path(wd) / "log.full.lammps").exists()


def test_agent_s_arrete_au_nombre_max_de_tours(tmp_path):
    client = FakeClient([[("parse_log", {})]] * 3)
    wd = str(tmp_path / "agent")
    out = ga.run_agent("test", wd, max_turns=3, client=client, model="faux")
    assert out.startswith("Arret")
    assert read_trace(wd)[-1] == {"type": "stop", "reason": "max_turns"}


# --------------------------------------------------------------------------- #
# Integration LAMMPS reelle
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_integration_lammps_traction_sans_defaut(tmp_path):
    if not (shutil.which("lmp") or shutil.which("lmp_serial") or shutil.which("lmp_mpi")):
        pytest.skip("executable LAMMPS introuvable")
    if ga.find_potential("BNC.tersoff") is None:
        pytest.skip("BNC.tersoff introuvable")
    spec = {"structure": {"size_x_nm": 5, "size_y_nm": 5},
            "input": {"task": "tensile", "equil_steps": 3000, "strain_rate_per_ps": 0.005}}
    wd = tmp_path / "run"
    out = ga.run_spec(spec, str(wd))
    assert out["ok"] is True, out
    res = out["result"]
    lo, hi = ga.PLAUSIBLE["young_TPa_pristine"]
    assert lo <= res["young_TPa"] <= hi
    assert res["young_TPa"] == pytest.approx(0.956, abs=0.05)      # reference LAMMPS 22Jul2025
    assert res["fit_r2"] > 0.9
    assert out["warnings"] == []
    traj = (wd / "traj.full.lammpstrj").read_text()
    assert traj.count("ITEM: TIMESTEP") >= ga.TRAJ_FRAMES
