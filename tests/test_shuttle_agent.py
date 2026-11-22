"""Tests de shuttle_agent : garde-fous, verdicts, boucle d'agent. Sans LLM et sans LAMMPS (calculs simules)."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import shuttle_agent as sa  # noqa: E402

LENGTH = sa.fs.tube_length(sa.CELLS)


def fake_run(**over):
    run = {"ok": True, "link": ["R", "R", "-", "-", "L", "L", "L"], "z_A": [10.0, 8.0, 2.0, -5.0, -20.0, -38.0, -30.0],
           "c60_intact": True, "crossed": True, "final_location": "left tube", "simulated_ps": 8.0,
           "release_ps": 2.0, "arrival_ps": 3.4, "q_flight_e": -1.25, "max_kinetic_eV": 21.0, "charge_drift_e": 0.0}
    run.update(over)
    return run


@pytest.fixture
def bench(tmp_path, monkeypatch):
    calls = {"fly": []}
    monkeypatch.setattr(sa.fs, "well", lambda wd, cells: {"binding_eV": 3.2, "retention_max_nN": 0.62,
                                                           "retention_max_at_A": 1.4, "wall_gap_A": 3.24})
    monkeypatch.setattr(sa.fs, "threshold", lambda cells, gap, depth, well, tube=None: {
        "ok": True, "threshold_V": 14.4, "q_ball_flight_e_per_V": -0.082, "release_at_A": 3.5})
    monkeypatch.setattr(sa.fs, "relax_cap", lambda wd, cells: sa.fs.capped_tube(cells))

    def fly(wd, cells, gap, depth, volt, **kw):
        calls["fly"].append((volt, depth, kw["flexible"], kw["tube"] is not None))
        return fake_run(c60_intact=kw["flexible"])                # rigide : casse ; flexible : survit
    monkeypatch.setattr(sa.fs, "fly", fly)
    b = sa.ShuttleBench(str(tmp_path))
    b.calls = calls
    return b


# --------------------------------------------------------------------------- #
# Garde-fous
# --------------------------------------------------------------------------- #
def test_le_seuil_exige_le_puits(bench):
    out = bench.ejection_threshold(0.0, True)
    assert out["ok"] is False and "measure_well" in out["errors"][0]


def test_vol_refuse_sans_seuil_pour_le_meme_montage(bench):
    bench.measure_well()
    bench.ejection_threshold(0.0, True)
    assert bench.fly(16.0, depth_A=0.0, capped=True)["ok"] is True
    for kwargs in ({"depth_A": 3.0, "capped": True}, {"depth_A": 0.0, "capped": False}):
        out = bench.fly(16.0, **kwargs)
        assert out["ok"] is False and "ejection_threshold" in out["errors"][0]
    assert len(bench.calls["fly"]) == 1                             # les vols refuses n'ont rien lance


@pytest.mark.parametrize("kwargs", [{"voltage_V": 100.0}, {"voltage_V": 16.0, "steps": 500_000},
                                    {"voltage_V": 16.0, "depth_A": 99.0}, {"voltage_V": 16.0, "flexible": "oui"},
                                    {"voltage_V": True}])
def test_vol_refuse_hors_bornes(bench, kwargs):
    bench.measure_well()
    bench.ejection_threshold(0.0, True)
    out = bench.fly(**kwargs)
    assert out["ok"] is False and bench.calls["fly"] == []


def test_budget_de_vols(bench):
    bench.measure_well()
    bench.ejection_threshold(0.0, True)
    for _ in range(sa.MAX_FLIGHTS):
        assert bench.fly(16.0)["ok"] is True
    out = bench.fly(16.0)
    assert out["ok"] is False and "budget" in out["errors"][0]
    assert len(bench.calls["fly"]) == sa.MAX_FLIGHTS


def test_dispatch_outil_inconnu_et_arguments_invalides(bench):
    assert sa.dispatch(bench, "write_lammps_input", "{}")["ok"] is False
    assert sa.dispatch(bench, "fly", "{pas du json")["ok"] is False
    assert sa.dispatch(bench, "fly", json.dumps({"tension": 16}))["ok"] is False
    assert bench.calls["fly"] == []


def test_le_schema_annonce_les_bornes_du_code():
    props = {t["function"]["name"]: t["function"]["parameters"]["properties"] for t in sa.TOOLS}
    for bound in ("voltage_V", "depth_A", "steps"):
        lo, hi = sa.BOUNDS[bound]
        assert props["fly"][bound]["minimum"] == lo and props["fly"][bound]["maximum"] == hi
    assert sa.BOUNDS["steps"][1] <= sa.fs.BOUNDS["steps"][1]         # jamais plus large que le module


# --------------------------------------------------------------------------- #
# Verdicts
# --------------------------------------------------------------------------- #
def test_verdicts():
    v = lambda capped=True, **over: sa.verdict(fake_run(**over), capped, LENGTH + 7.1)[0]  # noqa: E731
    assert v() == "crossed_intact"
    assert v(c60_intact=False) == "c60_destroyed_after_crossing"
    assert v(c60_intact=False, crossed=False) == "c60_destroyed"
    assert v(crossed=False, link=["R"] * 7, z_A=[10.0] * 7) == "not_ejected_within_run"
    # detache mais encore dans l'ecart quand le run s'arrete : ce n'est pas un echec, c'est un vol coupe
    assert v(crossed=False, link=["R"] * 5 + ["-", "-"], z_A=[10, 10, 10, 9, 8, 5, 1]) == "in_flight_at_end_of_run"
    assert v(capped=False, final_location="lost") == "c60_lost_through_open_end"
    # arrive dans le tube mais le run s'arrete avant qu'il soit stoppe (calotte ou sortie arriere non vues)
    assert v(z_A=[10.0, 8.0, 2.0, -5.0, -12.0, -15.0, -18.0]) == "crossed_but_stop_not_simulated"
    assert v(capped=False, z_A=[10.0, 8.0, 2.0, -5.0, -12.0, -15.0, -18.0]) == "crossed_but_stop_not_simulated"


def test_compte_des_traversees():
    run = fake_run(link=["R", "-", "L", "L", "-", "R", "-", "L"], z_A=[10, 0, -20, -44, 0, 20, 0, -20])
    assert sa.verdict(run, True, LENGTH + 7.1)[2] == 3


def test_rapport_plus_basse_tension_reussie_et_hypotheses(bench):
    assert bench.report()["ok"] is False
    bench.measure_well()
    bench.ejection_threshold(0.0, True)
    bench.fly(15.0, flexible=False)                                 # casse
    bench.fly(18.0, flexible=True)
    bench.fly(16.0, flexible=True)
    rep = bench.report()
    assert [f["verdict"] for f in rep["flights"]] == ["c60_destroyed_after_crossing", "crossed_intact", "crossed_intact"]
    assert rep["lowest_voltage_crossed_intact_V"] == 16.0           # le vol casse a 15 V ne compte pas
    text = " ".join(rep["hypotheses"])
    assert "1 fs" in text and "25 fs" in text and "3.2 eV" in text and "emission de champ" in text


# --------------------------------------------------------------------------- #
# Boucle d'agent avec un faux client
# --------------------------------------------------------------------------- #
class FakeClient:
    """Rejoue une liste d'appels d'outils, puis rend une reponse finale."""

    def __init__(self, script):
        self.script, self.seen = list(script), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, messages, tools):
        self.seen.append(messages[-1])
        if not self.script:
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="fini", tool_calls=None))])
        name, args = self.script.pop(0)
        call = SimpleNamespace(id=f"c{len(self.seen)}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[call]))])


def test_boucle_agent_trace_et_refus_visibles_du_modele(tmp_path, bench, monkeypatch):
    client = FakeClient([("fly", {"voltage_V": 16}), ("measure_well", {}), ("ejection_threshold", {"depth_A": 0, "capped": True}),
                         ("fly", {"voltage_V": 16, "flexible": True}), ("report", {})])
    out = sa.run_agent("fais traverser le C60", str(tmp_path / "run"), client=client, model="faux")
    assert out == "fini"
    recs = [json.loads(l) for l in (tmp_path / "run" / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["type"] for r in recs] == ["start", "tool", "tool", "tool", "tool", "tool", "final"]
    assert recs[1]["result"]["ok"] is False                         # le vol premature a ete refuse...
    assert json.loads(client.seen[1]["content"])["ok"] is False     # ...et le modele a vu le refus
    assert recs[4]["result"]["verdict"] == "crossed_intact"
    assert recs[5]["result"]["lowest_voltage_crossed_intact_V"] == 16


def test_arret_au_nombre_maximum_de_tours(tmp_path, bench):
    client = FakeClient([("measure_well", {})] * 10)
    out = sa.run_agent("x", str(tmp_path / "run"), max_turns=3, client=client, model="faux")
    assert out.startswith("Arret")
