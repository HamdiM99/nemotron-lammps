# graphene_agent (v0)

Un agent LLM qui pilote LAMMPS sur une feuille de graphène : relaxation et traction
uniaxiale (module de Young).

## Principe : le LLM propose, le code vérifie

- Le modèle n'a accès qu'à 4 outils bornés : `build_structure`, `write_input`, `run`, `parse_log`.
- Il n'écrit jamais d'input LAMMPS : il choisit des paramètres, le code génère le fichier.
- Toute validation est déterministe : bornes codées (`BOUNDS`, `MAX_ATOMS`), LAMMPS
  lui-même, puis contrôles de plausibilité physique (`PLAUSIBLE`).
- Un run `full` est refusé tant qu'un run `smoke` (200 pas) n'a pas réussi sur le même input.
- Chaque appel d'outil et la réponse finale sont journalisés dans `<workdir>/trace.jsonl`.

## Installation

Python 3.10+ et LAMMPS avec le paquet MANYBODY (potentiels `BNC.tersoff`, `CH.airebo`).

Linux / macOS :

    conda install -c conda-forge lammps
    pip install openai pytest

Windows (conda-forge ne fournit pas LAMMPS pour win-64) :

    pip install lammps openai pytest
    winget install Microsoft.msmpi

Le runtime Microsoft MPI est indispensable : sans `msmpi.dll`, `lmp.exe` quitte sans
message (code 0xC0000135) et l'agent rapporte « aucune sortie thermo dans le log ».

Le fichier de potentiel est cherché dans `$LAMMPS_POTENTIALS`, dans le paquet Python
`lammps`, à côté de l'exécutable `lmp`, puis dans le dossier courant.

## Usage

Pipeline déterministe, sans LLM (pour vérifier LAMMPS et les garde-fous) :

    python graphene_agent.py --spec spec.json

Agent (API compatible OpenAI, endpoint NVIDIA par défaut) :

    export NVIDIA_API_KEY=...        # PowerShell : $env:NVIDIA_API_KEY = "..."
    python graphene_agent.py "module de Young d'une feuille 5x5 nm avec 1 % de lacunes à 300 K" --workdir runs/essai

Variables optionnelles : `NEMOTRON_MODEL` (défaut `nvidia/nemotron-3.5-super-vl-120b-a12b`),
`LLM_BASE_URL`. La clé ne doit figurer dans aucun fichier.

Fichiers produits dans le dossier de travail : `graphene.data`, `in.<mode>.lammps`,
`log.<mode>.lammps`, `traj.<mode>.lammpstrj` (environ 100 images de la traction, pour
la visualisation) et `trace.jsonl`.

Tests :

    pytest             # garde-fous, parse_thermo, boucle agent avec faux modèle (quelques secondes)
    pytest -m slow     # intégration LAMMPS réelle (~30 s)

Interface 3D (three.js) d'un run de traction fait par l'agent, et vidéo :

    python viewer/build_viewer.py runs/essai    # écrit viewer/graphene_viewer.html et docs/index.html (pages autonomes)
    python viewer/make_video.py                 # écrit viewer/graphene_traction.mp4 (Edge/Chrome headless + imageio-ffmpeg)

`docs/index.html` est la page à partager : sur GitHub, activer Pages (Settings > Pages, branche `main`,
dossier `/docs`) la publie à l'adresse `https://<compte>.github.io/<dépôt>/`.

Banc d'essai (30 cas dans `bench/cases.json`, juge déterministe sur `trace.jsonl`) :

    python bench/run_bench.py                   # reprend là où il s'est arrêté ; attend et réessaie sur les 429
    python bench/run_bench.py --judge-only      # rejuge les traces existantes sans appeler le modèle

Sorties : `bench/results.csv`, `bench/summary.txt`, `bench/runs/<id>/`.

## Résultats de référence

Tersoff (`BNC.tersoff`), 300 K, LAMMPS 22Jul2025.

| Cas | Paramètres | Module de Young | Origine |
|---|---|---|---|
| 5x5 nm sans défaut | dt 1 fs, 3000 pas d'équilibration, 0,005 /ps, déformation max 2 % | 0,956 TPa (R² 0,996) | `--spec`, rejoué sur Windows (Update 4) |
| 5x5 nm, 2 % de lacunes | mêmes paramètres | 0,798 TPa | validation initiale, non rejouée ici |
| 5x5 nm, 1 % de lacunes, graine 42 | dt 0,1 fs, 10 000 pas, 0,01 /ps, déformation max 5 % (choisis par le modèle) | 0,825 TPa (R² 0,999) | agent réel Nemotron |

Repère expérimental : E ≈ 1,0 TPa pour le graphène sans défaut, a ≈ 2,46 Å. Tersoff
donne ici a = 2,492 Å.

## Limites

- **Potentiel classique** (Tersoff ou AIREBO) : les valeurs dépendent du potentiel et ne
  valent pas un calcul ab initio. L'épaisseur conventionnelle de 3,35 Å entre dans la contrainte.
- **Pas de propriétés électroniques** : gap, structure de bandes, dopage électronique sont
  hors de portée ; l'agent doit le dire sans avancer de chiffre.
- **Pas de conductivité thermique** dans le v0.
- **Régime linéaire seulement** : la déformation est bornée à 5 %, il n'y a donc ni
  résistance à la rupture ni rupture simulée.
- La plage de plausibilité du module de Young ne s'applique qu'aux feuilles sans défaut ;
  avec des lacunes, seul le R² de l'ajustement est contrôlé.
- Vitesses de déformation de dynamique moléculaire (10⁻⁴ à 10⁻² /ps), très supérieures à
  l'expérience ; feuilles de 1 à 20 nm, 20 000 atomes au plus, un seul tirage de lacunes par graine.
- Le modèle choisit librement les paramètres non précisés, dans les bornes : lire les
  hypothèses de la réponse et `trace.jsonl` plutôt que le chiffre seul.
