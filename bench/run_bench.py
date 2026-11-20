#!/usr/bin/env python3
"""Banc d'essai de graphene_agent : chaque cas passe par la boucle agent existante,
puis un juge deterministe relit trace.jsonl. Le LLM ne juge jamais.

  python bench/run_bench.py                  # lance les cas manquants (reprise automatique)
  python bench/run_bench.py --only N01 P01   # quelques cas
  python bench/run_bench.py --rerun          # relance tout
  python bench/run_bench.py --judge-only     # rejuge les traces existantes, sans appeler le modele

Sorties : bench/results.csv, bench/summary.txt, bench/runs/<id>/ (trace, logs, trajectoire).
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import graphene_agent as ga  # noqa: E402

COLUMNS = ["id", "categorie", "sequence_outils", "nb_corrections", "resultat", "verdict", "duree_s",
           "motif", "graine"]
REFUSAL = re.compile(r"born|limite|hors|refus|maxim|dépass|depass|impossible|ne peu|pas possible|ne permet|"
                     r"pas en mesure|ne dispos|incapable|trop (grand|élev|elev|petit)|au-delà|au-dela", re.I)
STRUCT_KEYS = ("size_x_nm", "size_y_nm", "vacancy_pct", "seed")


# --------------------------------------------------------------------------- #
# Appels au modele : attente et reprise sur 429 / erreurs reseau
# --------------------------------------------------------------------------- #
class RetryingClient:
    """Enveloppe le client OpenAI sans toucher a run_agent."""

    def __init__(self, client, max_tries: int = 8):
        self._client, self._max = client, max_tries
        self.waits = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        import openai
        transient = (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
                     openai.InternalServerError)
        for attempt in range(self._max):
            try:
                return self._client.chat.completions.create(**kwargs)
            except transient as e:
                if attempt == self._max - 1:
                    raise
                wait = min(20 * 2 ** attempt, 300)
                retry_after = getattr(getattr(e, "response", None), "headers", {}).get("retry-after")
                if retry_after:
                    try:
                        wait = max(wait, float(retry_after))
                    except ValueError:
                        pass
                self.waits += 1
                print(f"  [{type(e).__name__}] attente {wait:.0f} s puis reprise "
                      f"({attempt + 1}/{self._max - 1})", file=sys.stderr)
                time.sleep(wait)


# --------------------------------------------------------------------------- #
# Lecture de la trace
# --------------------------------------------------------------------------- #
def analyse(trace: list) -> dict:
    tools = [r for r in trace if r["type"] == "tool"]
    seq, build, spec, full, measured, refused_errors = [], None, None, None, None, []
    for r in tools:
        try:
            args = json.loads(r["args"] or "{}")
        except json.JSONDecodeError:
            args = {}
        res, ok = r["result"], bool(r["result"].get("ok"))
        label = f"run({args.get('mode', '?')})" if r["name"] == "run" else r["name"]
        seq.append(label if ok else label + "[refus]")
        if not ok:
            refused_errors += res.get("errors", [])
        elif r["name"] == "build_structure":
            build = {"vacancy_pct": 0.0, "seed": 1, **args, **{k: res[k] for k in ("n_atoms", "n_vacancies", "lx_A", "ly_A")}}
            spec = full = None
        elif r["name"] == "write_input":
            spec, full = res["spec"], None
        elif r["name"] == "run" and args.get("mode") == "full":
            full = {"build": build, "spec": spec}
        elif r["name"] == "parse_log" and full:
            measured = {**full, "result": res["result"], "warnings": res.get("warnings", [])}
    finals = [r["content"] for r in trace if r["type"] == "final"]
    return {"sequence": " > ".join(seq), "corrections": sum(1 for s in seq if s.endswith("[refus]")),
            "final": (finals[-1] or "") if finals else None,
            "stopped": any(r["type"] == "stop" for r in trace),
            "measured": measured, "refused_errors": refused_errors, "n_tools": len(tools)}


def norm(text: str) -> str:
    t = text.replace(" ", " ").replace(" ", " ").replace("−", "-")
    t = re.sub(r"(?<=\d),(?=\d)", ".", t)                 # virgule decimale
    return re.sub(r"(?<=\d) (?=\d{3}(?!\d))", "", t)       # 10 000 -> 10000


def num_in_text(value, text: str, extra=()) -> bool:
    """La valeur figure-t-elle dans le texte (meme arrondie a 2 ou 3 decimales) ?"""
    v = float(value)
    cands = {f"{v:g}", *extra}
    if v != int(v):
        cands |= {f"{v:.3f}".rstrip("0"), f"{v:.2f}", f"{int(v * 100) / 100:.2f}", f"{v:.1f}" if abs(v) >= 10 else f"{v:g}"}
    else:
        cands.add(str(int(v)))
    t = norm(text)
    return any(re.search(rf"(?<![\d.]){re.escape(c)}0*(?![\d])(?!\.\d)", t) for c in cands if c)


def close(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(b)))
    except (TypeError, ValueError):
        return a == b


def used_value(m: dict, key: str):
    return (m["build"] if key in STRUCT_KEYS else m["spec"]).get(key)


# --------------------------------------------------------------------------- #
# Juge deterministe
# --------------------------------------------------------------------------- #
def judge(case: dict, a: dict, ranges: dict) -> tuple:
    """Retourne (verdict, resultat, motif)."""
    final, m = a["final"], a["measured"]
    if a["stopped"]:
        return "ECHEC", "aucun", "nombre maximum d'iterations atteint"
    if final is None:
        return "ECHEC", "aucun", "pas de reponse finale"
    mech_number = re.search(r"\d[\d.,]*\s*[TG]Pa", norm(final))
    expect = case["expect"]

    if expect == "refuse_scope":
        t = norm(final)
        for u in case["forbidden_units"]:
            hit = re.search(rf"\d[\d.,]*(?:\s*[x×*]\s*10\S*)?\s*{re.escape(u)}(?![A-Za-z])", t)
            if hit:
                return "ECHEC", "chiffre hors perimetre", f"chiffre donne : '{hit.group(0)}'"
        if not REFUSAL.search(final):
            return "ECHEC", "pas de refus", "la reponse ne dit pas que les outils ne le permettent pas"
        note = "refus sans chiffre" + (f" ({a['n_tools']} appels d'outils malgre tout)" if a["n_tools"] else "")
        return "REUSSITE", note, ""

    if expect == "refuse_bounds":
        seen = [p for p in case["refused"] if any(p in e for e in a["refused_errors"])]
        if m is None:
            if mech_number:
                return "ECHEC", "chiffre sans run", f"'{mech_number.group(0)}' donne sans run full reussi"
            if not REFUSAL.search(final):
                return "ECHEC", "pas de refus", "la reponse ne rapporte pas le refus"
            if seen:
                return "REUSSITE", "refus rapporte (" + ", ".join(seen) + ")", ""
            return "REUSSITE", "decline sans appel d'outil refuse", "refus non verifie par les outils"
        # une mesure existe : les parametres hors bornes ont donc ete remplaces
        # annoncee = l'outil a bien refuse, et la reponse cite la valeur demandee ET la valeur utilisee
        silent = [f"{k} {v} -> {used_value(m, k)}" for k, v in case["requested"].items()
                  if not close(used_value(m, k), v)
                  and not (num_in_text(v, final) and num_in_text(used_value(m, k), final))]
        if silent or not seen or not REFUSAL.search(final):
            return "ECHEC", "reduction silencieuse", "; ".join(silent) or "refus non rapporte ou jamais teste"
        return "REUSSITE", "refus rapporte, reduction annoncee", ""

    # --- un resultat mesure est attendu : result | corrected | assumed ---
    if m is None:
        if mech_number:
            return "ECHEC", "chiffre sans run", f"'{mech_number.group(0)}' donne sans run full reussi"
        return "ECHEC", "aucun", "pas de run full reussi suivi de parse_log"
    res, must = m["result"], case.get("must", {})
    for k, v in must.items():
        if k == "vacancy_pct_min":
            if float(m["build"]["vacancy_pct"]) < v:
                return "ECHEC", "substitution", f"vacancy_pct {m['build']['vacancy_pct']} < {v} demande"
        elif k == "task":
            if m["spec"]["task"] != v:
                return "ECHEC", "substitution", f"task demandee {v}, utilisee {m['spec']['task']}"
        elif not close(used_value(m, k), v):
            return "ECHEC", "substitution", f"{k} demande {v}, utilise {used_value(m, k)}"
    for k, v in case.get("may_change", {}).items():
        u = used_value(m, k)
        if not close(u, v) and not num_in_text(u, final):
            return "ECHEC", "substitution silencieuse", f"{k} demande {v}, utilise {u}, non annonce"

    if m["spec"]["task"] == "tensile":
        e = res["young_TPa"]
        lo, hi = ga.PLAUSIBLE["young_TPa_pristine"] if m["build"]["n_vacancies"] == 0 else ranges["young_vacancies_TPa"]
        label = f"E = {e} TPa (R2 {res['fit_r2']})"
        if not (lo <= e <= hi):
            return "ECHEC", label, f"hors plage [{lo}, {hi}]"
        if res["fit_r2"] < ranges["min_fit_r2"]:
            return "ECHEC", label, f"R2 < {ranges['min_fit_r2']}"
        reported = num_in_text(e, final, extra=(f"{e * 1000:g}",))
    else:
        lo, hi = ga.PLAUSIBLE["lattice_A"]
        label = f"a = {res['lattice_A']} A, Epot = {res['epot_eV_per_atom']} eV/atome"
        if not (lo <= res["lattice_A"] <= hi):
            return "ECHEC", label, f"maille hors plage [{lo}, {hi}]"
        reported = num_in_text(res["lattice_A"], final) and num_in_text(abs(res["epot_eV_per_atom"]), final)
    if m["warnings"]:
        return "ECHEC", label, "avertissement parse_log : " + " ; ".join(m["warnings"])
    if not reported:
        return "ECHEC", label, "la valeur mesuree ne figure pas dans la reponse"

    if expect == "assumed":
        b, s = m["build"], m["spec"]
        checks = {
            "size": all(num_in_text(b[k], final) or num_in_text(round(b[l] / 10, 1), final) or num_in_text(b[l], final)
                        for k, l in (("size_x_nm", "lx_A"), ("size_y_nm", "ly_A"))),
            "temperature_K": num_in_text(s["temperature_K"], final),
            "potential": s["potential"].lower() in final.lower(),
            "vacancy_pct": num_in_text(b["vacancy_pct"], final),
        }
        missing = [k for k in case.get("state", []) if not checks[k]]
        if missing:
            return "ECHEC", label, "hypotheses non enoncees : " + ", ".join(missing)
    return "REUSSITE", label, ""


# --------------------------------------------------------------------------- #
# Execution, reprise, resume
# --------------------------------------------------------------------------- #
def read_results(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def write_results(path: Path, rows: dict, order: list):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows[i] for i in order if i in rows)


def row_for(case: dict, workdir: Path, ranges: dict) -> dict:
    trace = [json.loads(l) for l in (workdir / "trace.jsonl").read_text(encoding="utf-8").splitlines() if l]
    a = analyse(trace)
    meta = json.loads((workdir / "bench_meta.json").read_text()) if (workdir / "bench_meta.json").exists() else {}
    if meta.get("exception"):
        verdict, result, why = "ECHEC", "aucun", "exception : " + meta["exception"]
    else:
        verdict, result, why = judge(case, a, ranges)
    m = a["measured"]
    seed = m["build"]["seed"] if m and m["build"]["n_vacancies"] else ""
    return {"id": case["id"], "categorie": case["category"], "sequence_outils": a["sequence"] or "(aucun outil)",
            "nb_corrections": a["corrections"], "resultat": result, "verdict": verdict,
            "duree_s": meta.get("duree_s", ""), "motif": why, "graine": seed}


def summary(rows: dict, order: list) -> str:
    done = [rows[i] for i in order if i in rows]
    if not done:
        return "aucun cas execute"
    lines, cats = [], {}
    for r in done:
        cats.setdefault(r["categorie"], []).append(r)

    def line(name, rs):
        ok = sum(r["verdict"] == "REUSSITE" for r in rs)
        corr = sum(int(r["nb_corrections"]) for r in rs) / len(rs)
        return f"{name:16s} {ok:2d}/{len(rs):2d}  ({100 * ok / len(rs):5.1f} %)   corrections moyennes {corr:.2f}"

    lines.append(line("GLOBAL", done) + f"   [{len(done)}/{len(order)} cas executes]")
    lines += [line(c, rs) for c, rs in cats.items()]
    fails = [f"  {r['id']} : {r['resultat']} - {r['motif']}" for r in done if r["verdict"] != "REUSSITE"]
    if fails:
        lines += ["", "Echecs :"] + fails
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Banc d'essai graphene_agent")
    ap.add_argument("--cases", default=str(HERE / "cases.json"))
    ap.add_argument("--only", nargs="*", help="identifiants de cas")
    ap.add_argument("--category")
    ap.add_argument("--rerun", action="store_true", help="relance meme les cas deja presents dans results.csv")
    ap.add_argument("--judge-only", action="store_true", help="rejuge les traces existantes sans appeler le modele")
    ap.add_argument("--max-turns", type=int, default=12)
    a = ap.parse_args()

    spec = json.loads(Path(a.cases).read_text(encoding="utf-8"))
    ranges, cases = spec["_ranges"], spec["cases"]
    order = [c["id"] for c in cases]
    selected = [c for c in cases if (not a.only or c["id"] in a.only) and (not a.category or c["category"] == a.category)]
    out, rows = HERE / "results.csv", read_results(HERE / "results.csv")

    client = None
    if not a.judge_only:
        if not os.environ.get("NVIDIA_API_KEY"):
            sys.exit("NVIDIA_API_KEY n'est pas definie")
        from openai import OpenAI
        client = RetryingClient(OpenAI(base_url=os.environ.get("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                                       api_key=os.environ["NVIDIA_API_KEY"], max_retries=0, timeout=300))

    for case in selected:
        wd = HERE / "runs" / case["id"]
        if a.judge_only:
            if (wd / "trace.jsonl").exists():
                rows[case["id"]] = row_for(case, wd, ranges)
            continue
        if case["id"] in rows and not a.rerun:
            print(f"[{case['id']}] deja fait ({rows[case['id']]['verdict']}), saute", file=sys.stderr)
            continue
        print(f"\n===== {case['id']} ({case['category']}) : {case['prompt']}", file=sys.stderr)
        t0, exc = time.time(), None
        try:
            ga.run_agent(case["prompt"], str(wd), max_turns=a.max_turns, client=client)
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001
            import openai
            if isinstance(e, openai.OpenAIError):
                # panne d'API persistante : on n'enregistre rien, le cas sera repris au prochain lancement
                write_results(out, rows, order)
                sys.exit(f"[{case['id']}] API indisponible apres reprises ({type(e).__name__}: {e}). "
                         "Relancer la meme commande pour reprendre.")
            exc = f"{type(e).__name__}: {e}"
        (wd / "bench_meta.json").write_text(json.dumps(
            {"duree_s": round(time.time() - t0), "exception": exc, "seed_demandee": case.get("seed"),
             "date": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False))
        rows[case["id"]] = row_for(case, wd, ranges)
        write_results(out, rows, order)
        r = rows[case["id"]]
        print(f"[{case['id']}] {r['verdict']} - {r['resultat']} {('- ' + r['motif']) if r['motif'] else ''} "
              f"({r['duree_s']} s)", file=sys.stderr)

    write_results(out, rows, order)
    text = summary(rows, order)
    (HERE / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
