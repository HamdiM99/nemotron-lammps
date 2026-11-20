#!/usr/bin/env python3
"""Assemble une video de l'interface : une capture d'ecran par image, puis ffmpeg.

  python viewer/make_video.py            # ecrit viewer/graphene_traction.mp4

Utilise Edge ou Chrome en mode headless sur viewer/graphene_viewer.html?capture (partie haute
seulement : vue 3D, resultat, courbe) et le binaire ffmpeg du paquet imageio-ffmpeg.
La feuille tourne legerement pendant que la trajectoire defile.
"""
import argparse
import math
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            "msedge", "google-chrome", "chromium"]


def browser() -> str:
    for b in BROWSERS:
        p = b if Path(b).is_file() else shutil.which(b)
        if p:
            return p
    sys.exit("Edge ou Chrome introuvable")


def plan(n_traj: int, hold_start: int, hold_end: int, az0: float, az1: float) -> list:
    """Liste (image de trajectoire, azimut) : pause, traction, pause ; rotation douce sur l'ensemble."""
    seq = [0] * hold_start + list(range(n_traj)) + [n_traj - 1] * hold_end
    total = len(seq)
    return [(f, az0 + (az1 - az0) * (1 - math.cos(math.pi * k / (total - 1))) / 2) for k, f in enumerate(seq)]


def shot(exe, url, out, size, tmp):
    cmd = [exe, "--headless=new", "--disable-gpu-sandbox", "--enable-unsafe-swiftshader", "--hide-scrollbars",
           "--force-device-scale-factor=1", f"--window-size={size}", "--virtual-time-budget=6000",
           f"--user-data-dir={tmp}", f"--screenshot={out}", url]
    subprocess.run(cmd, capture_output=True, timeout=120)
    return out.exists()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frames", type=int, default=101, help="nombre d'images de la trajectoire")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--size", default="1360,1000")
    ap.add_argument("--only", type=int, help="ne capturer que cette image (essai)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    exe, frames_dir = browser(), HERE / "video_frames"
    frames_dir.mkdir(exist_ok=True)
    base = (HERE / "graphene_viewer.html").as_uri()
    steps = plan(a.frames, 10, 22, -0.68, -0.38)
    jobs = [(k, f, az) for k, (f, az) in enumerate(steps) if a.only is None or k == a.only]

    def work(job):
        k, f, az = job
        out = frames_dir / f"f{k:04d}.png"
        ok = out.exists() or shot(exe, f"{base}?capture&f={f}&az={az:.4f}", out, a.size, frames_dir / f"profile{k % a.workers}")
        return k, ok

    # un profil de navigateur par fil : deux instances ne partagent jamais le meme dossier
    with ThreadPoolExecutor(1 if a.only is not None else a.workers) as ex:
        groups = [[j for j in jobs if j[0] % a.workers == w] for w in range(a.workers)]
        results = [r for rs in ex.map(lambda g: [work(j) for j in g], groups) for r in rs]
    missing = sorted(k for k, ok in results if not ok)
    print(f"{len(results) - len(missing)}/{len(results)} images capturees dans {frames_dir}")
    if missing:
        sys.exit(f"images manquantes : {missing[:10]}")
    if a.only is not None:
        sys.exit(0)

    import imageio_ffmpeg
    out = HERE / "graphene_traction.mp4"
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-framerate", str(a.fps), "-i", str(frames_dir / "f%04d.png"),
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
           "-movflags", "+faststart", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stderr[-800:])
    print(f"{out} ({out.stat().st_size // 1024} ko, {len(steps) / a.fps:.1f} s)")
