"""Muziekjes - achtergrondmuziek eruit halen en namaken.

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


def naar_wav(pad: str) -> str:
    """Haal het geluid uit elk audio- of videobestand (mp4, mov, m4a, ...) met ffmpeg."""
    import subprocess
    tmp = OUT_DIR / "tmp"
    tmp.mkdir(exist_ok=True)
    doel = tmp / (Path(pad).stem[:40] + ".wav")
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", pad, "-vn",
                        "-ac", "2", "-ar", "44100", str(doel)], capture_output=True, text=True)
    if r.returncode != 0 or not doel.exists():
        raise gr.Error("Kon geen geluid uit dit bestand halen. Is het een audio- of videobestand?")
    return str(doel)


def naar_mp3(pad, progress=gr.Progress()):
    """Video (of ander audiobestand) omzetten naar mp3 (192 kbps)."""
    import subprocess
    if not pad:
        raise gr.Error("Upload eerst een video.")
    progress(0.2, desc="Omzetten naar mp3...")
    doel = _stamp(Path(pad).stem[:40]).with_suffix(".mp3")
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", pad, "-vn",
                        "-c:a", "libmp3lame", "-b:a", "192k", str(doel)],
                       capture_output=True, text=True)
    if r.returncode != 0 or not doel.exists():
        raise gr.Error("Kon geen geluid uit dit bestand halen. Is het een video- of audiobestand?")
    return str(doel), gr.update(value=str(doel), visible=True)


def pc_opnemen(seconden, progress=gr.Progress()):
    """Neem op wat de pc op dit moment afspeelt (Windows loopback van de standaard speaker)."""
    import warnings
    import soundcard as sc
    try:  # Gradio draait functies in een aparte thread; Windows-audio heeft daar COM nodig
        import ctypes
        ctypes.windll.ole32.CoInitializeEx(None, 0)
    except Exception:
        pass
    warnings.simplefilter("ignore", category=Warning)
    sr, seconden = 44100, int(seconden)
    spk = sc.default_speaker()
    mic = sc.get_microphone(id=str(spk.name), include_loopback=True)
    stukken = []
    with mic.recorder(samplerate=sr, channels=2) as rec:
        for s in range(seconden):
            progress((s + 1) / seconden, desc=f"Opnemen... {s + 1}/{seconden} sec")
            stukken.append(rec.record(numframes=sr))
    data = np.concatenate(stukken)
    if np.abs(data).max() < 1e-4:
        raise gr.Error("Er was niets te horen. Speelt er geluid af op de pc (en staat het volume aan)?")
    p = _stamp("pc-opname")
    sf.write(p, data, sr)
    return str(p), str(p)


def mic_gebruiken(opname, progress=None):
    if not opname:
        raise gr.Error("Neem eerst iets op met de microfoon.")
    return opname


def scheid(bestand, progress=gr.Progress()):
    """Eerste (beste) versie: htdemucs, ongewijzigd opgeslagen."""
    if not bestand:
        raise gr.Error("Upload eerst een nummer of video.")
    from demucs.apply import apply_model

    progress(0.02, desc="Geluid uit bestand halen...")
    bestand = naar_wav(bestand)
    progress(0.05, desc="Model laden...")
    model = get_demucs()
    sr = model.samplerate

    progress(0.15, desc="Audio inlezen...")
    wav, _ = librosa.load(bestand, sr=sr, mono=False)
    Path(bestand).unlink(missing_ok=True)
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


def _sleutel() -> str:
    """Afgeleide sleutel om ingelogd te blijven (het wachtwoord zelf wordt nergens bewaard)."""
    import hashlib
    return hashlib.sha256(("muziekjes:" + WACHTWOORD).encode()).hexdigest()


SLEUTEL = _sleutel()


def _klopt(waarde: str, verwacht: str) -> bool:
    import hmac
    return hmac.compare_digest((waarde or "").strip().encode(), verwacht.encode())


def _scherm(ok, melding=""):
    return ok, gr.update(visible=not ok), gr.update(visible=ok), melding


def inloggen(invoer):
    ok = _klopt(invoer, WACHTWOORD)
    return (*_scherm(ok, "" if ok else "❌ Verkeerd wachtwoord, probeer opnieuw."),
            SLEUTEL if ok else "")


def automatisch_inloggen(onthouden, request: gr.Request):
    """Bij openen/verversen: sleutel uit de browser of uit de link (?t=...) van de site."""
    if onthouden == "uitgelogd":  # bewust uitgelogd: niet automatisch weer inloggen
        return (*_scherm(False), onthouden)
    via_link = request.query_params.get("t", "") if request else ""
    ok = _klopt(via_link, SLEUTEL) or _klopt(onthouden, SLEUTEL)
    return (*_scherm(ok), SLEUTEL if ok else onthouden)


def uitloggen():
    return (*_scherm(False, "Je bent uitgelogd."), "uitgelogd")


def alleen_ingelogd(functie):
    """Login zit in de sessie op de server (geen cookies nodig, werkt op elke telefoon)."""
    def wrapper(ok, *args, progress=gr.Progress()):
        if not ok:
            raise gr.Error("Log eerst in.")
        return functie(*args, progress=progress)
    return wrapper


def _download_knoppen(*paden):
    """Zet downloadknoppen aan zodra er een resultaat is."""
    uit = [gr.update(value=p, visible=bool(p)) for p in paden]
    return uit[0] if len(uit) == 1 else uit


def _sporen_downloads(achtergrond, *sporen):
    return (_download_knoppen(achtergrond),
            gr.update(value=[s for s in sporen if s] or None, visible=any(sporen)))


def _dl(label):
    return gr.DownloadButton(f"⬇️ Download {label}", visible=False, variant="secondary")


with gr.Blocks(title="Muziekjes") as app:
    gr.Markdown("# 🎵 Muziekjes")
    ingelogd = gr.State(False)
    onthouden = gr.BrowserState("", storage_key="muziekjes_sleutel")

    with gr.Column(visible=True) as login_scherm:
        ww_invoer = gr.Textbox(label="Wachtwoord", type="password")
        login_knop = gr.Button("Inloggen", variant="primary")
        login_melding = gr.Markdown()

    with gr.Column(visible=False) as hoofd_scherm:
        gr.Markdown("Haal de achtergrondmuziek uit een nummer en maak er een nieuwe versie "
                    f"van in dezelfde stijl. *(draait op: {DEVICE})*")

        with gr.Tab("1. Achtergrond eruit halen"):
            inp = gr.File(label="Nummer of video (mp3, wav, m4a, mp4, mov, ...)",
                          file_types=["audio", "video", ".mp4", ".mov", ".m4a", ".m4v"],
                          type="filepath")
            knop1 = gr.Button("Zang eruit halen", variant="primary")
            achter = gr.Audio(label="Achtergrond (zonder zang)", type="filepath")
            dl_achter = _dl("achtergrond")
            with gr.Accordion("Losse sporen", open=False):
                drums = gr.Audio(label="Drums", type="filepath")
                bas = gr.Audio(label="Bas", type="filepath")
                overig = gr.Audio(label="Overige instrumenten", type="filepath")
                zang = gr.Audio(label="Zang", type="filepath")
                dl_sporen = gr.File(label="Download losse sporen", file_count="multiple",
                                    visible=False)

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
            uit, dl_uit = [], []
            for i in (1, 2, 3):
                uit.append(gr.Audio(label=f"Versie {i}", type="filepath"))
                dl_uit.append(_dl(f"versie {i}"))

        with gr.Tab("Opnemen"):
            gr.Markdown("### 💻 Geluid van de pc\nNeemt op wat de pc nu afspeelt "
                        "(YouTube, Spotify, ...). Zet eerst het geluid aan op de pc, druk dan "
                        "op opnemen. Loopt door, ook als je telefoon op slot gaat.")
            seconden = gr.Slider(5, 300, value=30, step=5, label="Hoe lang opnemen (sec)")
            knop_pc = gr.Button("⏺ Pc-geluid opnemen", variant="primary")
            pc_opname = gr.Audio(label="Pc-opname", type="filepath")
            dl_pc = _dl("pc-opname")
            gr.Markdown("### 📱 Microfoon (telefoon)\nNeemt op via de microfoon, bijvoorbeeld "
                        "muziek uit je speaker. Let op: laat de app open staan, op de "
                        "achtergrond of met je scherm op slot stopt de iPhone de opname. "
                        "Tip: de schermopname van je iPhone werkt wél op de achtergrond; die "
                        "mp4 kun je in stap 1 uploaden.")
            mic_opname = gr.Audio(label="Microfoon-opname", sources=["microphone"], type="filepath")
            knop_mic = gr.Button("Gebruik deze opname in stap 1")
            gr.Markdown("*Na het opnemen staat de opname automatisch klaar in stap 1.*")

        with gr.Tab("MP4 → MP3"):
            gr.Markdown("Zet een video om naar een mp3-bestand.")
            video = gr.File(label="Video (mp4, mov, ...)",
                            file_types=["video", "audio", ".mp4", ".mov", ".m4v", ".m4a"],
                            type="filepath")
            knop3 = gr.Button("Omzetten naar MP3", variant="primary")
            mp3_speler = gr.Audio(label="MP3", type="filepath")
            dl_mp3 = _dl("MP3")

        uitlog_knop = gr.Button("Uitloggen", size="sm", variant="secondary")

    # inloggen / ingelogd blijven
    login_uit = [ingelogd, login_scherm, hoofd_scherm, login_melding, onthouden]
    login_knop.click(inloggen, ww_invoer, login_uit)
    ww_invoer.submit(inloggen, ww_invoer, login_uit)
    app.load(automatisch_inloggen, onthouden, login_uit)
    uitlog_knop.click(uitloggen, None, login_uit)

    # acties + downloadknoppen
    knop1.click(alleen_ingelogd(scheid), [ingelogd, inp],
                [achter, drums, bas, overig, zang, voorbeeld]).then(
        _sporen_downloads, [achter, drums, bas, overig, zang], [dl_achter, dl_sporen])
    knop2.click(alleen_ingelogd(namaken), [ingelogd, voorbeeld, beschrijving, start, duur,
                                           creativiteit, versies, ontruis2], uit).then(
        _download_knoppen, uit, dl_uit)
    knop_pc.click(alleen_ingelogd(pc_opnemen), [ingelogd, seconden], [pc_opname, inp]).then(
        _download_knoppen, pc_opname, dl_pc)
    knop_mic.click(alleen_ingelogd(mic_gebruiken), [ingelogd, mic_opname], inp)
    knop3.click(alleen_ingelogd(naar_mp3), [ingelogd, video], [mp3_speler, dl_mp3])


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
