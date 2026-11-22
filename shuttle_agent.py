#!/usr/bin/env python3
"""
shuttle_agent.py - le LLM pilote la navette a fullerene de fullerene_shuttle.py.

Meme principe que graphene_agent : LE LLM PROPOSE, LE CODE VERIFIE.
  - quatre outils bornes ; le modele n'ecrit jamais d'input LAMMPS ni de charges ;
  - un vol est refuse tant que le seuil statique n'a pas ete calcule pour le meme montage (equivalent du
    "smoke avant full") ;
  - nombre de vols limite par session ; chaque resultat porte un verdict et des hypotheses ecrits par le code.

Usage :
  python shuttle_agent.py "Trouve la tension minimale qui fait traverser le C60 sans le casser." --workdir runs_fs_agent
La cle est lue dans la variable d'environnement NVIDIA_API_KEY ; elle n'est ecrite nulle part.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import fullerene_shuttle as fs
import graphene_agent as ga
import nanotube_lab as nt

CELLS, GAP_A = 12, 20.0          # montage fixe : deux tubes (10,10) de 12 mailles, bouches a 2 nm
MAX_FLIGHTS = 6                  # budget de dynamique par session
UPDATE = 25                      # iterations entre deux recalculs des charges
THREADS = 2
# Bornes vues par le modele (plus etroites que celles du module pour le nombre de pas : budget de calcul).
BOUNDS = {
    "depth_A": fs.BOUNDS["depth_A"],
    "voltage_V": fs.BOUNDS["voltage_V"],
    "steps": (2000, 12_000),
}


def check(name: str, value, errors: list):
    lo, hi = BOUNDS[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not (lo <= value <= hi):
        errors.append(f"{name}={value} hors bornes [{lo}, {hi}]")


def verdict(run: dict, capped: bool, length: float) -> tuple:
    """(verdict, a atteint la calotte du tube d'arrivee, nombre de traversees) a partir d'un resultat de fs.fly.

    "crossed_intact" exige que le C60 ait ete ARRETE dans le tube d'arrivee (il y rebrousse chemin) en restant
    intact. Un vol coupe avant ce moment n'est ni une reussite ni un echec : le verdict le dit."""
    links, z = run["link"], run["z_A"]
    n_cross = sum(1 for a, b in zip(links, links[1:]) if a == "-" and b in ("L", "R"))
    # le centre du C60 s'arrete a environ 7 A du sommet de la calotte (rayon 3,55 A + ecart de van der Waals) ;
    # a moins de 9 A, il a touche le fond
    reached = capped and min(z) < -(GAP_A / 2 + length - 9.0)
    arrive = next((k for k in range(1, len(links)) if links[k] == "L" and links[k - 1] == "-"), None)
    turned_back, low = False, math.inf
    for v in (z[arrive:] if arrive is not None else []):         # apres l'arrivee : est-il reparti en arriere de plus de 2 A ?
        low = min(low, v)
        turned_back = turned_back or v > low + 2.0
    if not run["c60_intact"]:
        return ("c60_destroyed_after_crossing" if run["crossed"] else "c60_destroyed"), reached, n_cross
    if run["final_location"] == "lost":
        return "c60_lost_through_open_end", reached, n_cross
    if not run["crossed"]:
        return ("in_flight_at_end_of_run" if links[-1] == "-" else "not_ejected_within_run"), reached, n_cross
    if not turned_back:
        return "crossed_but_stop_not_simulated", reached, n_cross
    return "crossed_intact", reached, n_cross


class ShuttleBench:
    """Etat d'une session : puits mesure, seuils calcules, vols effectues. Chaque outil renvoie un dict."""

    def __init__(self, workdir: str = "runs_fs_agent"):
        self.dir = Path(workdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.well = None
        self.thresholds = {}         # (depth, capped) -> resultat de fs.threshold
        self.flights = []
        self._tube = None

    def _tube_for(self, capped: bool):
        if not capped:
            return None
        if self._tube is None:
            self._tube = fs.relax_cap(self.dir, CELLS)
        return self._tube

    # ---- outil 1 ----
    def measure_well(self) -> dict:
        if self.well is None:
            self.well = fs.well(self.dir, CELLS)
        w = self.well
        return {"ok": True, "binding_eV": w["binding_eV"], "retention_max_nN": w["retention_max_nN"],
                "retention_max_at_A": w["retention_max_at_A"], "warnings": fs.plausibility({"well": w})}

    # ---- outil 2 ----
    def ejection_threshold(self, depth_A: float = 0.0, capped: bool = True) -> dict:
        errors = []
        check("depth_A", depth_A, errors)
        if not isinstance(capped, bool):
            errors.append("capped doit etre true ou false")
        if self.well is None:
            errors.append("appelle measure_well d'abord : le seuil a besoin de la force de retenue")
        if errors:
            return {"ok": False, "errors": errors}
        res = fs.threshold(CELLS, GAP_A, float(depth_A), self.well, tube=self._tube_for(capped))
        if not res.get("ok"):
            return res
        self.thresholds[(float(depth_A), capped)] = res
        out = {"ok": True, "depth_A": depth_A, "capped": capped, "threshold_V": res.get("threshold_V"),
               "charge_at_release_e_per_V": res["q_ball_flight_e_per_V"], "release_at_A": res["release_at_A"]}
        if res.get("threshold_V") is None:
            out["note"] = "aucun seuil : la traction electrostatique ne depasse jamais la retenue sur ce trajet"
        return out

    # ---- outil 3 ----
    def fly(self, voltage_V: float, depth_A: float = 0.0, capped: bool = True, flexible: bool = False,
            steps: int = 8000) -> dict:
        errors = []
        check("voltage_V", voltage_V, errors)
        check("depth_A", depth_A, errors)
        check("steps", steps, errors)
        for name, v in (("capped", capped), ("flexible", flexible)):
            if not isinstance(v, bool):
                errors.append(f"{name} doit etre true ou false")
        if not errors and (float(depth_A), capped) not in self.thresholds:
            errors.append(f"vol refuse : calcule d'abord ejection_threshold(depth_A={depth_A}, capped={str(capped).lower()})")
        if len(self.flights) >= MAX_FLIGHTS:
            errors.append(f"budget epuise : {MAX_FLIGHTS} vols par session")
        if errors:
            return {"ok": False, "errors": errors}
        name = f"fly{len(self.flights) + 1}"
        tube = self._tube_for(capped)
        run = fs.fly(self.dir, CELLS, GAP_A, float(depth_A), float(voltage_V), steps=int(steps), update=UPDATE,
                     name=name, threads=THREADS, tube=tube, flexible=flexible)
        if not run.get("ok"):
            return run
        length = float(tube[:, 2].max()) if capped else fs.tube_length(CELLS)
        v, reached, n_cross = verdict(run, capped, length)
        thr = self.thresholds[(float(depth_A), capped)].get("threshold_V")
        out = {"ok": True, "flight": name, "voltage_V": voltage_V, "depth_A": depth_A, "capped": capped, "flexible": flexible,
               "steps": steps, "simulated_ps": run["simulated_ps"], "static_threshold_V": thr, "verdict": v,
               "crossings": n_cross, "reached_far_cap": reached, "c60_intact": run["c60_intact"],
               "release_ps": run["release_ps"], "arrival_ps": run["arrival_ps"], "charge_in_flight_e": run["q_flight_e"],
               "max_kinetic_eV": run["max_kinetic_eV"], "final_location": run["final_location"],
               "warnings": fs.plausibility({name: run})}
        self.flights.append(out)
        return out

    # ---- outil 4 ----
    def report(self) -> dict:
        if not self.flights:
            return {"ok": False, "errors": ["aucun vol effectue"]}
        ok = [f for f in self.flights if f["verdict"] == "crossed_intact"]
        best = min(ok, key=lambda f: f["voltage_V"]) if ok else None
        return {"ok": True, "flights": [{k: f[k] for k in ("flight", "voltage_V", "depth_A", "capped", "flexible",
                                                           "verdict", "crossings", "max_kinetic_eV")} for f in self.flights],
                "lowest_voltage_crossed_intact_V": best["voltage_V"] if best else None,
                "hypotheses": self._assumptions(), "warnings": sorted({w for f in self.flights for w in f["warnings"]})}

    def _assumptions(self) -> list:
        """Hypotheses ecrites par le code, unites comprises : le modele les recopie sans les convertir."""
        w = self.well or {}
        return [
            f"montage : deux nanotubes (10,10) de {fs.tube_length(CELLS) / 10:.2f} nm, bouches ecartees de {GAP_A / 10:.1f} nm, un C60",
            f"potentiel : AIREBO avec van der Waals, plus Coulomb ; pas de temps {nt.DT_PS * 1000:g} fs",
            f"charges : methode des moments, recalculees toutes les {UPDATE * nt.DT_PS * 1000:g} fs ; contact electrique sous {fs.CONTACT_A:g} A",
            f"puits du C60 dans le tube : {w.get('binding_eV')} eV ; retenue maximale {w.get('retention_max_nN')} nN",
            "temperature initiale 0 K ; un seul vol par jeu de parametres, pas de statistique",
            "non modelise : emission de champ, effet tunnel, dissipation dans un support",
            "le seuil statique est une borne haute du seuil dynamique pour un depart au repos",
        ]


def _num(bound: str, desc: str, kind: str = "number") -> dict:
    lo, hi = BOUNDS[bound]
    return {"type": kind, "minimum": lo, "maximum": hi, "description": f"{desc} ; bornes [{lo:g}, {hi:g}]"}


_DEPTH = "enfoncement initial du centre du C60 derriere la bouche du tube de depart, en A (0 = a la bouche)"
_CAPPED = {"type": "boolean", "description": "true : chaque tube est ferme au bout arriere par une calotte ; false : bout arriere ouvert"}
TOOLS = [
    ga._fn("measure_well", "Mesure le puits de van der Waals du C60 dans un tube : energie d'encapsulation (eV) et force "
           "de retenue maximale a la bouche (nN). A appeler en premier.", {}),
    ga._fn("ejection_threshold", "Calcul statique, rapide : tension (V) a partir de laquelle la traction electrostatique "
           "depasse la retenue sur tout le trajet de sortie, et charge prise par le C60 (e par volt). "
           "Obligatoire avant tout vol avec les memes depth_A et capped.",
           {"depth_A": _num("depth_A", _DEPTH), "capped": _CAPPED}),
    ga._fn("fly", "Dynamique moleculaire : le C60 part au repos du tube de droite, tubes a +V/2 et -V/2. "
           f"Renvoie un verdict. Couteux : {MAX_FLIGHTS} vols au plus par session.",
           {"voltage_V": _num("voltage_V", "tension entre les deux tubes, en V"),
            "depth_A": _num("depth_A", _DEPTH), "capped": _CAPPED,
            "flexible": {"type": "boolean", "description": "false : tubes parfaitement rigides ; true : tubes libres de "
                         "vibrer et de se deformer, tenus par un anneau au milieu (calcul plus lent)"},
            "steps": _num("steps", "nombre de pas de 1 fs (1000 pas = 1 ps)", "integer")}, ["voltage_V"]),
    ga._fn("report", "Tableau des vols de la session, plus basse tension ayant donne une traversee sans casse, "
           "hypotheses et avertissements. A appeler avant de repondre.", {}),
]

SYSTEM = """Tu pilotes une navette a fullerene avec quatre outils : un C60 doit passer d'un nanotube a l'autre sous tension.
Ordre : measure_well, ejection_threshold, fly, report.
Si un outil renvoie ok=false, lis les erreurs, corrige et recommence.
Un vol n'est une reussite que si son verdict est "crossed_intact". Lis les autres verdicts et adapte les parametres :
chaque option de fly correspond a un choix physique decrit dans l'outil.
Tu ne peux mesurer que ce que les outils permettent. Si la demande sort de ce cadre, dis-le au lieu d'improviser.
Dans la reponse finale, donne le resultat, recopie telles quelles les hypotheses renvoyees par report
(champ "hypotheses", unites comprises, sans les convertir) et tous les avertissements. N'invente aucun chiffre."""


def dispatch(bench: ShuttleBench, name: str, raw_args: str) -> dict:
    """Seuls les 4 outils declares sont appelables ; tout argument invalide revient comme erreur."""
    if name not in {t["function"]["name"] for t in TOOLS}:
        return {"ok": False, "errors": [f"outil inconnu : {name}"]}
    try:
        return getattr(bench, name)(**json.loads(raw_args or "{}"))
    except (TypeError, json.JSONDecodeError) as e:
        return {"ok": False, "errors": [f"arguments invalides : {e}"]}
    except RuntimeError as e:                                    # echec LAMMPS : rendu au modele, pas masque
        return {"ok": False, "errors": [f"echec du calcul : {str(e)[:300]}"]}


def run_agent(prompt: str, workdir: str, max_turns: int = 16, client=None, model: str | None = None) -> str:
    if client is None:
        from openai import OpenAI
        client = OpenAI(base_url=os.environ.get("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                        api_key=os.environ["NVIDIA_API_KEY"])
    model = model or os.environ.get("NEMOTRON_MODEL", "nvidia/nemotron-3.5-super-vl-120b-a12b")
    bench = ShuttleBench(workdir)
    trace = bench.dir / "trace.jsonl"
    trace.write_text("")

    def log(**rec):
        with trace.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    log(type="start", prompt=prompt, model=model)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    for _ in range(max_turns):
        msg = client.chat.completions.create(model=model, messages=messages, tools=TOOLS).choices[0].message
        if not msg.tool_calls:
            log(type="final", content=msg.content)
            return msg.content
        messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
            for c in msg.tool_calls]})
        for call in msg.tool_calls:
            result = dispatch(bench, call.function.name, call.function.arguments)
            print(f"[outil] {call.function.name}({call.function.arguments}) -> "
                  f"{json.dumps(result, ensure_ascii=False)[:400]}", file=sys.stderr, flush=True)
            log(type="tool", name=call.function.name, args=call.function.arguments, result=result)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=False)})
    log(type="stop", reason="max_turns")
    return "Arret : nombre maximum de tours atteint sans resultat."


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Agent LLM pour la navette a fullerene")
    ap.add_argument("prompt")
    ap.add_argument("--workdir", default="runs_fs_agent")
    ap.add_argument("--max-turns", type=int, default=16)
    a = ap.parse_args()
    if not os.environ.get("NVIDIA_API_KEY"):
        sys.exit("NVIDIA_API_KEY absente de l'environnement : arret.")
    print(run_agent(a.prompt, a.workdir, a.max_turns))
