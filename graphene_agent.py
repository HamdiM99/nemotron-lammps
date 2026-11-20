#!/usr/bin/env python3
"""
graphene_agent.py - v0 : un agent LLM qui pilote LAMMPS sur une feuille de graphene.

Principe : le LLM propose, le code verifie.
  - Le modele ne voit que 4 outils bornes : build_structure, write_input, run, parse_log.
  - Il n'ecrit jamais de fichier LAMMPS brut : il choisit des parametres, le code genere l'input.
  - Toute validation est deterministe (regles codees + LAMMPS lui-meme), jamais le LLM.
  - Un run complet est refuse tant qu'un run court ("smoke") n'a pas reussi sur le meme input.

Taches supportees :
  - relax   : relaxation, energie par atome et parametre de maille
  - tensile : traction uniaxiale, module de Young (pente contrainte/deformation)

Usage :
  # avec le LLM (API compatible OpenAI, par defaut l'endpoint NVIDIA)
  export NVIDIA_API_KEY=nvapi-...
  # modele par defaut : nvidia/nemotron-3.5-super-vl-120b-a12b (modifiable via NEMOTRON_MODEL)
  python graphene_agent.py "module de Young d'une feuille 5x5 nm avec 1 % de lacunes a 300 K"

  # sans LLM : execute le pipeline deterministe sur une spec JSON (pour tester LAMMPS seul)
  python graphene_agent.py --spec spec.json

Dependances : LAMMPS (executable `lmp`), et `pip install openai` pour le mode LLM.
"""
import argparse
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Constantes physiques et bornes (les garde-fous vivent ici, pas dans le LLM)
# --------------------------------------------------------------------------- #
ACC = 1.42            # distance C-C initiale (A)
THICKNESS = 3.35      # epaisseur conventionnelle du graphene (A), pour la contrainte
LZ = 20.0             # hauteur de boite (A) : vide au-dessus et en dessous de la feuille
MASS_C = 12.011

BOUNDS = {
    "size_nm": (1.0, 20.0),
    "vacancy_pct": (0.0, 10.0),
    "temperature_K": (1.0, 3000.0),
    "timestep_fs": (0.05, 2.0),          # au-dela, l'integration du carbone diverge
    "equil_steps": (500, 200_000),
    "strain_rate_per_ps": (1e-4, 1e-2),
    "max_strain": (0.005, 0.05),         # regime lineaire pour le module de Young
}
MAX_ATOMS = 20_000        # budget PC ; au-dela il faut un GPU ou un cluster
RUN_TIMEOUT_S = {"smoke": 120, "full": 3600}
SMOKE_STEPS = 200
TRAJ_FRAMES = 100         # nombre d'images visees dans traj.<mode>.lammpstrj

POTENTIALS = {
    "tersoff": ("BNC.tersoff", "pair_style tersoff\npair_coeff * * \"{path}\" C"),
    "airebo": ("CH.airebo", "pair_style airebo 3.0 0 0\npair_coeff * * \"{path}\" C"),
}
# Plages plausibles : un run qui se termine sans erreur peut etre faux.
PLAUSIBLE = {
    "young_TPa_pristine": (0.8, 1.3),
    "lattice_A": (2.40, 2.56),
    "epot_eV_per_atom": (-8.2, -6.8),
}


def find_potential(filename: str) -> Path | None:
    """Cherche le fichier de potentiel : $LAMMPS_POTENTIALS, paquet Python lammps, dossier courant."""
    dirs = []
    if os.environ.get("LAMMPS_POTENTIALS"):
        dirs.append(Path(os.environ["LAMMPS_POTENTIALS"]))
    try:
        import lammps  # noqa: WPS433
        root = Path(lammps.__file__).parent
        dirs += [root / "share/lammps/potentials"]
        dirs += [p for p in root.parent.parent.glob("dist-packages/lammps/share/lammps/potentials")]
    except Exception:
        pass
    lmp = shutil.which("lmp")
    if lmp:
        dirs.append(Path(lmp).resolve().parent.parent / "share/lammps/potentials")
    dirs.append(Path.cwd())
    for d in dirs:
        if (d / filename).is_file():
            return d / filename
    return None


def check_bounds(name: str, value: float, errors: list):
    lo, hi = BOUNDS[name]
    if not isinstance(value, (int, float)) or not (lo <= value <= hi):
        errors.append(f"{name}={value} hors bornes [{lo}, {hi}]")


# --------------------------------------------------------------------------- #
# Les 4 outils
# --------------------------------------------------------------------------- #
class Workbench:
    """Etat d'une etude : un dossier de travail et les 4 outils exposes au modele."""

    def __init__(self, workdir: str = "run"):
        self.dir = Path(workdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.structure = None     # infos de la structure courante
        self.spec = None          # parametres de l'input courant
        self.smoke_ok = False     # le run court a-t-il reussi sur cet input ?
        self.full_ok = False

    # -- 1. structure ------------------------------------------------------- #
    def build_structure(self, size_x_nm: float, size_y_nm: float,
                        vacancy_pct: float = 0.0, seed: int = 1) -> dict:
        errors = []
        check_bounds("size_nm", size_x_nm, errors)
        check_bounds("size_nm", size_y_nm, errors)
        check_bounds("vacancy_pct", vacancy_pct, errors)
        if errors:
            return {"ok": False, "errors": errors}

        # Maille orthorhombique a 4 atomes : x = armchair, y = zigzag
        ax, ay = 3 * ACC, math.sqrt(3) * ACC
        nx, ny = max(2, round(size_x_nm * 10 / ax)), max(2, round(size_y_nm * 10 / ay))
        basis = [(0, 0), (ACC, 0), (1.5 * ACC, ay / 2), (2.5 * ACC, ay / 2)]
        atoms = [(i * ax + bx, j * ay + by, LZ / 2)
                 for i in range(nx) for j in range(ny) for bx, by in basis]
        if len(atoms) > MAX_ATOMS:
            return {"ok": False, "errors": [f"{len(atoms)} atomes > budget {MAX_ATOMS}"]}

        n_vac = round(len(atoms) * vacancy_pct / 100)
        rng = random.Random(seed)
        removed = set(rng.sample(range(len(atoms)), n_vac))
        atoms = [a for k, a in enumerate(atoms) if k not in removed]

        lines = ["graphene (genere par graphene_agent)", "",
                 f"{len(atoms)} atoms", "1 atom types", "",
                 f"0.0 {nx * ax:.6f} xlo xhi", f"0.0 {ny * ay:.6f} ylo yhi", f"0.0 {LZ} zlo zhi", "",
                 "Masses", "", f"1 {MASS_C}", "", "Atoms # atomic", ""]
        lines += [f"{k + 1} 1 {x:.6f} {y:.6f} {z:.6f}" for k, (x, y, z) in enumerate(atoms)]
        (self.dir / "graphene.data").write_text("\n".join(lines) + "\n")

        self.structure = {"n_atoms": len(atoms), "n_vacancies": n_vac, "nx": nx, "ny": ny,
                          "lx_A": round(nx * ax, 3), "ly_A": round(ny * ay, 3),
                          "vacancy_pct": vacancy_pct}
        self.spec, self.smoke_ok, self.full_ok = None, False, False
        return {"ok": True, **self.structure}

    # -- 2. input ----------------------------------------------------------- #
    def write_input(self, task: str, potential: str = "tersoff", temperature_K: float = 300.0,
                    timestep_fs: float = 1.0, equil_steps: int = 5000,
                    strain_rate_per_ps: float = 0.002, max_strain: float = 0.02) -> dict:
        errors = []
        if self.structure is None:
            errors.append("aucune structure : appeler build_structure d'abord")
        if task not in ("relax", "tensile"):
            errors.append(f"task={task!r} inconnue (relax | tensile)")
        if potential not in POTENTIALS:
            errors.append(f"potential={potential!r} inconnu ({' | '.join(POTENTIALS)})")
        check_bounds("temperature_K", temperature_K, errors)
        check_bounds("timestep_fs", timestep_fs, errors)
        check_bounds("equil_steps", equil_steps, errors)
        check_bounds("strain_rate_per_ps", strain_rate_per_ps, errors)
        check_bounds("max_strain", max_strain, errors)
        pot_path = find_potential(POTENTIALS[potential][0]) if potential in POTENTIALS else None
        if potential in POTENTIALS and pot_path is None:
            errors.append(f"fichier {POTENTIALS[potential][0]} introuvable "
                          "(definir LAMMPS_POTENTIALS ou le copier dans le dossier courant)")
        if errors:
            return {"ok": False, "errors": errors}

        self.spec = {"task": task, "potential": potential, "potential_path": str(pot_path),
                     "temperature_K": temperature_K, "timestep_fs": timestep_fs,
                     "equil_steps": int(equil_steps), "strain_rate_per_ps": strain_rate_per_ps,
                     "max_strain": max_strain}
        self.smoke_ok, self.full_ok = False, False
        self._render("full")
        warnings = []
        if timestep_fs > 1.0:
            warnings.append("pas de temps > 1 fs : risque de derive pour le carbone")
        if task == "tensile":
            self.spec["deform_steps"] = self._deform_steps()
        return {"ok": True, "spec": self.spec, "warnings": warnings}

    def _deform_steps(self) -> int:
        s = self.spec
        return int(s["max_strain"] / (s["strain_rate_per_ps"] * s["timestep_fs"] * 1e-3))

    def _render(self, mode: str) -> Path:
        """Genere l'input LAMMPS a partir de la spec. Le mode smoke raccourcit les runs."""
        s = self.spec
        dt = s["timestep_fs"] * 1e-3  # ps (units metal)
        equil = SMOKE_STEPS if mode == "smoke" else s["equil_steps"]
        thermo = 20 if mode == "smoke" else 50
        pair = POTENTIALS[s["potential"]][1].format(path=Path(s["potential_path"]).as_posix())
        txt = f"""# genere par graphene_agent ({mode}) - ne pas editer a la main
units metal
dimension 3
boundary p p p
atom_style atomic
read_data graphene.data
{pair}
neighbor 2.0 bin
thermo_style custom step temp pe ke etotal pxx pyy lx ly
thermo {thermo}
thermo_modify flush yes

# 1. relaxation des positions et de la boite dans le plan
fix br all box/relax x 0.0 y 0.0
min_style cg
minimize 1e-10 1e-10 5000 50000
unfix br
"""
        if s["task"] == "tensile":
            deform = SMOKE_STEPS if mode == "smoke" else self._deform_steps()
            txt += f"""
# 2. equilibration NPT dans le plan
reset_timestep 0
timestep {dt}
velocity all create {s['temperature_K']} 4928459 mom yes rot yes dist gaussian
fix eq all npt temp {s['temperature_K']} {s['temperature_K']} {100 * dt} x 0 0 {1000 * dt} y 0 0 {1000 * dt}
run {equil}
unfix eq

# 3. traction selon x, pression nulle selon y
reset_timestep 0
fix pull all npt temp {s['temperature_K']} {s['temperature_K']} {100 * dt} y 0 0 {1000 * dt}
fix def all deform 1 x erate {s['strain_rate_per_ps']} remap x
# trajectoire pour la visualisation (~{TRAJ_FRAMES} images)
dump traj all custom {max(1, deform // TRAJ_FRAMES)} traj.{mode}.lammpstrj id type x y z
dump_modify traj sort id
run {deform}
"""
        else:
            txt += f"\n# structure relaxee pour la visualisation\nwrite_dump all custom traj.{mode}.lammpstrj id type x y z modify sort id\n"
        path = self.dir / f"in.{mode}.lammps"
        path.write_text(txt)
        return path

    # -- 3. run ------------------------------------------------------------- #
    def run(self, mode: str = "smoke") -> dict:
        if self.spec is None:
            return {"ok": False, "errors": ["aucun input : appeler write_input d'abord"]}
        if mode not in ("smoke", "full"):
            return {"ok": False, "errors": ["mode inconnu (smoke | full)"]}
        if mode == "full" and not self.smoke_ok:
            return {"ok": False, "errors": ["run complet refuse : le run smoke doit reussir d'abord"]}
        lmp = shutil.which("lmp") or shutil.which("lmp_serial") or shutil.which("lmp_mpi")
        if lmp is None:
            return {"ok": False, "errors": ["executable LAMMPS (lmp) introuvable dans le PATH"]}

        inp = self._render(mode)
        log = f"log.{mode}.lammps"
        try:
            proc = subprocess.run([lmp, "-in", inp.name, "-log", log, "-screen", "none"],
                                  cwd=self.dir, capture_output=True, text=True,
                                  timeout=RUN_TIMEOUT_S[mode])
        except subprocess.TimeoutExpired:
            return {"ok": False, "errors": [f"budget de temps depasse ({RUN_TIMEOUT_S[mode]} s)"]}

        text = (self.dir / log).read_text(errors="replace") if (self.dir / log).exists() else ""
        errors = [l.strip() for l in (text + proc.stderr).splitlines() if l.startswith("ERROR")]
        if proc.returncode != 0 and not errors:
            errors.append(f"LAMMPS a quitte avec le code {proc.returncode}")
        errors += self._sanity(parse_thermo(text), mode)
        ok = not errors
        if mode == "smoke":
            self.smoke_ok = ok
        else:
            self.full_ok = ok
        out = {"ok": ok, "mode": mode, "errors": errors}
        if not ok:
            out["log_tail"] = "\n".join(text.splitlines()[-15:])
        return out

    def _sanity(self, blocks: list, mode: str) -> list:
        """Controles physiques communs a tout run : pas de NaN, energie et temperature saines."""
        if not blocks:
            return ["aucune sortie thermo dans le log"]
        errors = []
        n = self.structure["n_atoms"]
        for b in blocks:
            for row in b:
                if any(math.isnan(v) or math.isinf(v) for v in row.values()):
                    return ["valeurs NaN/inf : la simulation a diverge"]
        last = blocks[-1][-1]
        lo, hi = PLAUSIBLE["epot_eV_per_atom"]
        if not (lo <= last["PotEng"] / n <= hi):
            errors.append(f"energie potentielle {last['PotEng'] / n:.2f} eV/atome hors plage [{lo}, {hi}]")
        if self.spec["task"] == "tensile":
            t = self.spec["temperature_K"]
            temps = [r["Temp"] for r in blocks[-1][len(blocks[-1]) // 2:]]
            mean_t = sum(temps) / len(temps)
            # un run smoke est trop court pour thermaliser : on y cherche seulement l'emballement
            if mode == "smoke":
                if mean_t > 3 * t + 100:
                    errors.append(f"temperature {mean_t:.0f} K : emballement (consigne {t:.0f} K)")
            elif abs(mean_t - t) > max(0.25 * t, 30):
                errors.append(f"temperature moyenne {mean_t:.0f} K loin de la consigne {t:.0f} K")
        return errors

    # -- 4. analyse --------------------------------------------------------- #
    def parse_log(self) -> dict:
        if not self.full_ok:
            return {"ok": False, "errors": ["aucun run complet reussi a analyser"]}
        blocks = parse_thermo((self.dir / "log.full.lammps").read_text(errors="replace"))
        n, s, warnings = self.structure["n_atoms"], self.spec, []
        relaxed = blocks[0][-1]
        res = {"task": s["task"], "potential": s["potential"], "n_atoms": n,
               "epot_eV_per_atom": round(relaxed["PotEng"] / n, 4),
               "lattice_A": round(relaxed["Ly"] / self.structure["ny"], 4)}
        lo, hi = PLAUSIBLE["lattice_A"]
        if not (lo <= res["lattice_A"] <= hi):
            warnings.append(f"parametre de maille {res['lattice_A']} A hors plage [{lo}, {hi}]")

        if s["task"] == "tensile":
            rows = blocks[-1]
            lx0 = rows[0]["Lx"]
            strain = [(r["Lx"] - lx0) / lx0 for r in rows]
            # contrainte ramenee a l'epaisseur conventionnelle du graphene, bar -> GPa
            stress = [-r["Pxx"] * 1e-4 * LZ / THICKNESS for r in rows]
            slope, r2 = linfit(strain, stress)
            res.update({"young_TPa": round(slope / 1000, 3), "fit_r2": round(r2, 3),
                        "max_strain": round(strain[-1], 4), "temperature_K": s["temperature_K"]})
            if r2 < 0.9:
                warnings.append(f"ajustement lineaire mediocre (R2={r2:.2f}) : "
                                "bruit thermique eleve ou sortie du regime lineaire")
            lo, hi = PLAUSIBLE["young_TPa_pristine"]
            if self.structure["n_vacancies"] == 0 and not (lo <= res["young_TPa"] <= hi):
                warnings.append(f"module de Young {res['young_TPa']} TPa hors plage attendue "
                                f"[{lo}, {hi}] pour du graphene sans defaut")
        res["reference"] = "graphene sans defaut : E ~ 1.0 TPa (experience), a ~ 2.46 A"
        return {"ok": True, "result": res, "warnings": warnings}


# --------------------------------------------------------------------------- #
# Lecture du log
# --------------------------------------------------------------------------- #
def parse_thermo(text: str) -> list:
    """Retourne une liste de blocs thermo ; chaque bloc est une liste de dicts colonne -> valeur."""
    blocks, header, rows = [], None, []
    for line in text.splitlines():
        tok = line.split()
        if tok and tok[0] == "Step":
            header, rows = tok, []
        elif header and line.startswith("Loop time"):
            if rows:
                blocks.append(rows)
            header = None
        elif header and len(tok) == len(header):
            try:
                rows.append(dict(zip(header, map(float, tok))))
            except ValueError:
                pass
    if header and rows:   # run interrompu : on garde ce qui a ete ecrit
        blocks.append(rows)
    return blocks


def linfit(x: list, y: list) -> tuple:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    syy = sum((b - my) ** 2 for b in y)
    slope = sxy / sxx
    return slope, (sxy ** 2 / (sxx * syy) if syy else 0.0)


# --------------------------------------------------------------------------- #
# Boucle agent
# --------------------------------------------------------------------------- #
def _fn(name, desc, props, required=()):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": list(required)}}}


TOOLS = [
    _fn("build_structure", "Construit une feuille de graphene periodique et ecrit le fichier de donnees.",
        {"size_x_nm": {"type": "number"}, "size_y_nm": {"type": "number"},
         "vacancy_pct": {"type": "number", "description": "pourcentage d'atomes retires"},
         "seed": {"type": "integer"}}, ["size_x_nm", "size_y_nm"]),
    _fn("write_input", "Valide les parametres et genere l'input LAMMPS. "
        "tersoff ou airebo conviennent pour la mecanique ; aucun des deux ne donne de proprietes electroniques.",
        {"task": {"type": "string", "enum": ["relax", "tensile"]},
         "potential": {"type": "string", "enum": list(POTENTIALS)},
         "temperature_K": {"type": "number"}, "timestep_fs": {"type": "number"},
         "equil_steps": {"type": "integer"}, "strain_rate_per_ps": {"type": "number"},
         "max_strain": {"type": "number"}}, ["task"]),
    _fn("run", "Lance LAMMPS. 'smoke' = run court de validation, obligatoire avant 'full'.",
        {"mode": {"type": "string", "enum": ["smoke", "full"]}}, ["mode"]),
    _fn("parse_log", "Extrait les grandeurs du run complet et applique les controles de plausibilite.", {}),
]

SYSTEM = """Tu pilotes des simulations LAMMPS de graphene avec quatre outils.
Ordre : build_structure, write_input, run(smoke), run(full), parse_log.
Si un outil renvoie ok=false, lis les erreurs, corrige les parametres et recommence.
Tu ne peux mesurer que ce que les outils permettent : relaxation (energie, maille) et traction (module de Young).
Si la demande sort de ce cadre (proprietes electroniques, conductivite thermique...), dis-le au lieu d'improviser.
Dans la reponse finale, donne la valeur, les hypotheses (potentiel, taille, temperature, vitesse de deformation)
et tous les avertissements renvoyes par parse_log. N'invente aucun chiffre."""


def run_agent(prompt: str, workdir: str, max_turns: int = 12, client=None, model: str | None = None) -> str:
    if client is None:
        from openai import OpenAI
        client = OpenAI(base_url=os.environ.get("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                        api_key=os.environ["NVIDIA_API_KEY"])
    model = model or os.environ.get("NEMOTRON_MODEL", "nvidia/nemotron-3.5-super-vl-120b-a12b")
    bench = Workbench(workdir)
    trace = bench.dir / "trace.jsonl"     # journal des appels d'outils, relu par le banc d'essai
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
            {"id": c.id, "type": "function",
             "function": {"name": c.function.name, "arguments": c.function.arguments}}
            for c in msg.tool_calls]})
        for call in msg.tool_calls:
            result = dispatch(bench, call.function.name, call.function.arguments)
            print(f"[outil] {call.function.name}({call.function.arguments}) -> "
                  f"{json.dumps(result, ensure_ascii=False)[:300]}", file=sys.stderr)
            log(type="tool", name=call.function.name, args=call.function.arguments, result=result)
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(result, ensure_ascii=False)})
    log(type="stop", reason="max_turns")
    return "Arret : nombre maximum de tours atteint sans resultat."


def dispatch(bench: Workbench, name: str, raw_args: str) -> dict:
    """Seuls les 4 outils declares sont appelables ; tout argument invalide revient comme erreur."""
    if name not in {t["function"]["name"] for t in TOOLS}:
        return {"ok": False, "errors": [f"outil inconnu : {name}"]}
    try:
        return getattr(bench, name)(**json.loads(raw_args or "{}"))
    except (TypeError, json.JSONDecodeError) as e:
        return {"ok": False, "errors": [f"arguments invalides : {e}"]}


def run_spec(spec: dict, workdir: str) -> dict:
    """Pipeline deterministe sans LLM, pour tester LAMMPS et les garde-fous."""
    bench = Workbench(workdir)
    steps = [("build_structure", spec.get("structure", {})), ("write_input", spec.get("input", {})),
             ("run", {"mode": "smoke"}), ("run", {"mode": "full"}), ("parse_log", {})]
    out = {}
    for name, args in steps:
        out = getattr(bench, name)(**args)
        print(f"[{name}] {json.dumps(out, ensure_ascii=False)}", file=sys.stderr)
        if not out.get("ok"):
            break
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Agent LAMMPS graphene (v0)")
    ap.add_argument("prompt", nargs="?", help="demande en langage naturel")
    ap.add_argument("--spec", help="fichier JSON : execute le pipeline sans LLM")
    ap.add_argument("--workdir", default="run")
    a = ap.parse_args()
    if a.spec:
        print(json.dumps(run_spec(json.loads(Path(a.spec).read_text()), a.workdir), indent=2, ensure_ascii=False))
    elif a.prompt:
        print(run_agent(a.prompt, a.workdir))
    else:
        ap.error("donner un prompt ou --spec")
