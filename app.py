"""Muziek Kloon - achtergrondmuziek eruit halen en namaken.

Stap 1: Demucs haalt de zang weg -> instrumentale achtergrond + losse sporen.
Stap 2: MusicGen-Melody maakt nieuwe muziek die de melodie/stijl van de achtergrond volgt.
"""
import os
import time
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path(__file__).parent / "uitvoer"
OUT_DIR.mkdir(exist_ok=True)

_demucs = None
_musicgen = None


def _stamp(name: str) -> Path:
    return OUT_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{name}.wav"


# ---------------------------------------------------------------- stap 1 ---
def get_demucs():
    global _demucs
    if _demucs is None:
        from demucs.pretrained import get_model
        _demucs = get_model("htdemucs").to(DEVICE).eval()
    return _demucs


def scheid(bestand, progress=gr.Progress()):
    if not bestand:
        raise gr.Error("Upload eerst een nummer.")
    from demucs.apply import apply_model

    progress(0.05, desc="Model laden...")
    model = get_demucs()
    sr = model.samplerate

    progress(0.15, desc="Audio inlezen...")
    wav, _ = librosa.load(bestand, sr=sr, mono=False)
    if wav.ndim == 1:
        wav = np.stack([wav, wav])
    wav = torch.from_numpy(wav[:2]).float()
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)

    progress(0.3, desc="Zang en instrumenten scheiden (kan even duren)...")
    with torch.no_grad():
        sources = apply_model(model, wav[None], device=DEVICE, split=True,
                              overlap=0.25, progress=False)[0]
    sources = sources * ref.std() + ref.mean()
    stems = {name: sources[i].cpu().numpy() for i, name in enumerate(model.sources)}

    progress(0.9, desc="Opslaan...")
    stem = Path(bestand).stem
    paden = {}
    instrumentaal = sum(v for k, v in stems.items() if k != "vocals")
    for name, data in [("achtergrond", instrumentaal), *stems.items()]:
        p = _stamp(f"{stem}_{name}")
        sf.write(p, data.T, sr)
        paden[name] = str(p)

    return (paden["achtergrond"], paden["drums"], paden["bass"],
            paden["other"], paden["vocals"], paden["achtergrond"])


# ---------------------------------------------------------------- stap 2 ---
def get_musicgen():
    global _musicgen
    if _musicgen is None:
        from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration
        repo = "facebook/musicgen-melody"
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        processor = AutoProcessor.from_pretrained(repo)
        model = MusicgenMelodyForConditionalGeneration.from_pretrained(
            repo, torch_dtype=dtype).to(DEVICE).eval()
        _musicgen = (processor, model)
    return _musicgen


def namaken(bestand, beschrijving, start, duur, creativiteit, versies,
            progress=gr.Progress()):
    if not bestand:
        raise gr.Error("Kies een voorbeeld (bijv. de achtergrond uit stap 1).")
    progress(0.05, desc="MusicGen laden (eerste keer ~3 GB downloaden)...")
    processor, model = get_musicgen()
    sr = processor.feature_extractor.sampling_rate  # 32 kHz

    audio, _ = librosa.load(bestand, sr=sr, mono=True, offset=float(start),
                            duration=float(duur))
    if len(audio) < sr:
        raise gr.Error("Het gekozen stuk is te kort (startpunt voorbij het einde?).")

    tekst = beschrijving.strip() or "instrumental backing track, same style, high quality"
    inputs = processor(audio=[audio] * int(versies), sampling_rate=sr, text=[tekst] * int(versies),
                       padding=True, return_tensors="pt").to(DEVICE)
    if DEVICE == "cuda":
        inputs = {k: (v.half() if v.is_floating_point() else v) for k, v in inputs.items()}

    progress(0.3, desc="Nieuwe muziek genereren...")
    tokens = int(duur * model.config.audio_encoder.frame_rate)
    with torch.no_grad():
        out = model.generate(**inputs, do_sample=True, guidance_scale=3.0,
                             temperature=float(creativiteit), max_new_tokens=tokens)

    out_sr = model.config.audio_encoder.sampling_rate
    resultaten = []
    for i in range(int(versies)):
        p = _stamp(f"kloon_{i + 1}")
        sf.write(p, out[i, 0].float().cpu().numpy(), out_sr)
        resultaten.append(str(p))
    resultaten += [None] * (3 - len(resultaten))
    return resultaten


# -------------------------------------------------------------------- UI ---
with gr.Blocks(title="Muziek Kloon") as app:
    gr.Markdown("# 🎵 Muziek Kloon\nHaal de achtergrondmuziek uit een nummer en maak er "
                f"een nieuwe versie van in dezelfde stijl. *(draait op: {DEVICE})*")

    with gr.Tab("1. Achtergrond eruit halen"):
        inp = gr.Audio(label="Nummer (mp3, wav, ...)", type="filepath")
        knop1 = gr.Button("Zang eruit halen", variant="primary")
        achter = gr.Audio(label="Achtergrond (zonder zang)")
        with gr.Accordion("Losse sporen", open=False):
            drums = gr.Audio(label="Drums")
            bas = gr.Audio(label="Bas")
            overig = gr.Audio(label="Overige instrumenten")
            zang = gr.Audio(label="Zang")

    with gr.Tab("2. Nieuwe achtergrond maken"):
        gr.Markdown("Gebruikt het voorbeeld voor melodie en stijl. Na stap 1 staat de "
                    "achtergrond hier automatisch klaar; je kunt ook zelf iets uploaden.")
        voorbeeld = gr.Audio(label="Voorbeeld", type="filepath")
        beschrijving = gr.Textbox(
            label="Beschrijving (Engels werkt het best)",
            placeholder="bijv. chill lo-fi hip hop beat with piano and soft drums")
        with gr.Row():
            start = gr.Number(label="Start in voorbeeld (sec)", value=0, minimum=0)
            duur = gr.Slider(5, 30, value=15, step=1, label="Lengte (sec)")
        with gr.Row():
            creativiteit = gr.Slider(0.5, 1.5, value=1.0, step=0.05,
                                     label="Creativiteit (laag = dichter bij origineel)")
            versies = gr.Slider(1, 3, value=1, step=1, label="Aantal versies")
        knop2 = gr.Button("Nieuwe achtergrond maken", variant="primary")
        uit = [gr.Audio(label=f"Versie {i}") for i in (1, 2, 3)]

    knop1.click(scheid, inp, [achter, drums, bas, overig, zang, voorbeeld])
    knop2.click(namaken, [voorbeeld, beschrijving, start, duur, creativiteit, versies], uit)

def wachtwoord() -> str:
    """Lokaal wachtwoord (staat NIET op GitHub)."""
    bestand = Path(__file__).parent / "wachtwoord.txt"
    if not bestand.exists():
        import secrets
        bestand.write_text(secrets.token_urlsafe(9), encoding="utf-8")
    return bestand.read_text(encoding="utf-8").strip()


def publiceer_link(url: str):
    """Zet de nieuwe openbare link in docs/index.html en push naar GitHub Pages."""
    import subprocess
    root = Path(__file__).parent
    sjabloon = (root / "docs" / "sjabloon.html").read_text(encoding="utf-8")
    (root / "docs" / "index.html").write_text(sjabloon.replace("{{URL}}", url), encoding="utf-8")
    git = lambda *a: subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True)
    git("add", "docs/index.html")
    git("commit", "-m", "Nieuwe link naar de app")
    r = git("push")
    print("GitHub Pages bijgewerkt." if r.returncode == 0 else f"Push mislukt: {r.stderr}")


if __name__ == "__main__":
    ww = wachtwoord()
    print(f"\n  Inloggen: gebruiker 'muziek', wachtwoord '{ww}'\n")
    _, lokaal, openbaar = app.queue().launch(
        share=True, auth=("muziek", ww), inbrowser=True, prevent_thread_lock=True)
    if openbaar:
        print(f"  Openbare link: {openbaar}")
        publiceer_link(openbaar)
    app.block_thread()
