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


# --------------------------------------------------------- nabewerking ---
def afwerken(x: np.ndarray, sr: int, ontruisen: float = 0.0, normaliseren: bool = True):
    """Maak audio schoon: DC weg, ruis/tikken eruit, volume op peil, zachte randen.

    x: (kanalen, samples) of (samples,). ontruisen: 0 (uit) .. 1 (sterk).
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float32))
    x = x - x.mean(axis=1, keepdims=True)

    if ontruisen > 0:
        import noisereduce as nr
        x = nr.reduce_noise(y=x, sr=sr, stationary=False, prop_decrease=float(ontruisen),
                            time_mask_smooth_ms=64, freq_mask_smooth_hz=500,
                            n_fft=2048).astype(np.float32)

    piek = float(np.abs(x).max()) + 1e-9
    if normaliseren:
        rms = float(np.sqrt((x ** 2).mean())) + 1e-9
        gain = min(10 ** (-16 / 20) / rms, 0.89 / piek)  # ~-16 dB RMS, max -1 dBFS piek
    else:
        gain = min(1.0, 0.89 / piek)  # alleen oversturing voorkomen
    x *= gain

    fade = min(int(0.02 * sr), x.shape[1] // 4)
    if fade > 0:
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        x[:, :fade] *= ramp
        x[:, -fade:] *= ramp[::-1]
    return x


def opslaan(pad: Path, x: np.ndarray, sr: int):
    sf.write(pad, x.T if x.shape[0] <= 2 else x, sr, subtype="PCM_24")


# ---------------------------------------------------------------- stap 1 ---
def get_demucs():
    global _demucs
    if _demucs is None:
        from demucs.pretrained import get_model
        _demucs = get_model("htdemucs").to(DEVICE).eval()
    return _demucs


def scheid(bestand, progress=gr.Progress()):
    """Eerste (beste) versie: htdemucs, ongewijzigd opgeslagen."""
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
    stem = Path(bestand).stem[:40]
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
        # geluid decoderen in volle precisie: fp16 hier geeft ruis en tikjes
        model.audio_encoder.float()
        _musicgen = (processor, model)
    return _musicgen


def namaken(bestand, beschrijving, start, duur, creativiteit, versies, ontruisen=0.4,
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
        opslaan(p, afwerken(out[i, 0].float().cpu().numpy(), out_sr, float(ontruisen)), out_sr)
        resultaten.append(str(p))
    resultaten += [None] * (3 - len(resultaten))
    return resultaten


# -------------------------------------------------------------------- UI ---
def wachtwoord() -> str:
    """Lokaal wachtwoord (staat NIET op GitHub)."""
    bestand = Path(__file__).parent / "wachtwoord.txt"
    if not bestand.exists():
        import secrets
        bestand.write_text(secrets.token_urlsafe(9), encoding="utf-8")
    return bestand.read_text(encoding="utf-8").strip()


WACHTWOORD = wachtwoord()


def inloggen(invoer):
    import hmac
    ok = hmac.compare_digest((invoer or "").strip().encode(), WACHTWOORD.encode())
    return (ok, gr.update(visible=not ok), gr.update(visible=ok),
            "" if ok else "❌ Verkeerd wachtwoord, probeer opnieuw.")


def alleen_ingelogd(functie):
    """Login zit in de sessie op de server (geen cookies nodig, werkt op elke telefoon)."""
    def wrapper(ok, *args, progress=gr.Progress()):
        if not ok:
            raise gr.Error("Log eerst in.")
        return functie(*args, progress=progress)
    return wrapper


with gr.Blocks(title="Muziek Kloon") as app:
    gr.Markdown("# 🎵 Muziek Kloon")
    ingelogd = gr.State(False)

    with gr.Column(visible=True) as login_scherm:
        ww_invoer = gr.Textbox(label="Wachtwoord", type="password")
        login_knop = gr.Button("Inloggen", variant="primary")
        login_melding = gr.Markdown()

    with gr.Column(visible=False) as hoofd_scherm:
        gr.Markdown("Haal de achtergrondmuziek uit een nummer en maak er een nieuwe versie "
                    f"van in dezelfde stijl. *(draait op: {DEVICE})*")

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
            ontruis2 = gr.Slider(0, 1, value=0.4, step=0.05,
                                 label="Ruis en tikken verwijderen (0 = uit, 1 = sterk)")
            knop2 = gr.Button("Nieuwe achtergrond maken", variant="primary")
            uit = [gr.Audio(label=f"Versie {i}") for i in (1, 2, 3)]

    login_uit = [ingelogd, login_scherm, hoofd_scherm, login_melding]
    login_knop.click(inloggen, ww_invoer, login_uit)
    ww_invoer.submit(inloggen, ww_invoer, login_uit)
    knop1.click(alleen_ingelogd(scheid), [ingelogd, inp],
                [achter, drums, bas, overig, zang, voorbeeld])
    knop2.click(alleen_ingelogd(namaken), [ingelogd, voorbeeld, beschrijving, start, duur,
                                           creativiteit, versies, ontruis2], uit)


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
    print(f"\n  Wachtwoord voor de app: {WACHTWOORD}\n")
    _, lokaal, openbaar = app.queue().launch(
        share=True, inbrowser=True, prevent_thread_lock=True, max_file_size="200mb")
    if openbaar:
        print(f"  Openbare link: {openbaar}")
        publiceer_link(openbaar)
    app.block_thread()
