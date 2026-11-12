"""Video naar tekst: YouTube-link of eigen video -> transcript met pauzes, opbouw en samenvatting.

- yt-dlp haalt geluid + titel + hoofdstukken op
- faster-whisper (large-v3-turbo, GPU) maakt het transcript met tijd per woord
- pauzes = stiltes tussen woorden; lange pauzes worden een nieuwe alinea
- Ollama (lokaal LLM) beschrijft de opbouw en maakt een samenvatting
"""
import json
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np

OLLAMA = "http://127.0.0.1:11434"
OLLAMA_MODEL = "qwen3:14b"
WHISPER_MODEL = "large-v3-turbo"
SR = 16000


# ------------------------------------------------------------------ hulp ---
def tijd(sec: float) -> str:
    sec = int(round(sec))
    u, rest = divmod(sec, 3600)
    m, s = divmod(rest, 60)
    return f"{u}:{m:02d}:{s:02d}" if u else f"{m:02d}:{s:02d}"


def _audio_laden(pad: str) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", pad, "-vn", "-ac", "1",
                          "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


# --------------------------------------------------------------- ophalen ---
def youtube_ophalen(url: str, map_: Path) -> tuple[str, dict]:
    import yt_dlp
    map_.mkdir(parents=True, exist_ok=True)
    opts = {"format": "bestaudio/best", "outtmpl": str(map_ / "%(id)s.%(ext)s"),
            "quiet": True, "no_warnings": True, "noplaylist": True, "noprogress": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        pad = ydl.prepare_filename(info)
    meta = {"titel": info.get("title") or "", "kanaal": info.get("channel") or info.get("uploader") or "",
            "duur": info.get("duration") or 0, "link": info.get("webpage_url") or url,
            "datum": info.get("upload_date") or "",
            "hoofdstukken": [(c.get("start_time", 0), c.get("title", "")) for c in
                             (info.get("chapters") or [])]}
    return pad, meta


# ----------------------------------------------------------- transcript ---
def transcriberen(audio: np.ndarray, taal: str | None = None):
    import torch  # noqa: F401  laadt de CUDA/cuDNN-bestanden voor ctranslate2
    from faster_whisper import WhisperModel
    model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="int8_float16")
    segs, info = model.transcribe(audio, language=taal or None, word_timestamps=True,
                                  vad_filter=True, beam_size=5)
    woorden = [(w.start, w.end, w.word) for s in segs for w in (s.words or [])]
    del model  # videogeheugen vrijgeven voor de rest
    return woorden, info.language


def alinea_s(woorden, pauze_min: float, alinea_pauze: float, hoofdstuk_starts=()):
    """Maak alinea's: [{start, eind, tekst (met [pauze]-markeringen), pauzes:[...]}]."""
    alineas, huidig, pauzes = [], None, []
    vorige_eind = 0.0
    grenzen = sorted(t for t in hoofdstuk_starts if t > 0)
    for start, eind, woord in woorden:
        gat = start - vorige_eind
        nieuw_hoofdstuk = bool(grenzen) and start >= grenzen[0] - 0.3
        while grenzen and start >= grenzen[0] - 0.3:
            grenzen.pop(0)
        if huidig is None or gat >= alinea_pauze or nieuw_hoofdstuk:
            if huidig is not None:
                alineas.append(huidig)
            if gat >= pauze_min and (huidig is not None or start >= pauze_min):
                pauzes.append((vorige_eind, gat))
            huidig = {"start": start, "eind": eind, "tekst": "", "pauze_ervoor": gat}
        elif gat >= pauze_min:
            pauzes.append((vorige_eind, gat))
            huidig["tekst"] += f" [pauze {gat:.1f}s]"
        huidig["tekst"] += woord
        huidig["eind"] = eind
        vorige_eind = eind
    if huidig is not None:
        alineas.append(huidig)
    for a in alineas:
        a["tekst"] = " ".join(a["tekst"].split())
    return alineas, pauzes


# ------------------------------------------------------- opbouw (Ollama) ---
def ollama_beschikbaar() -> bool:
    try:
        with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=3) as r:
            namen = [m["name"] for m in json.load(r)["models"]]
        return OLLAMA_MODEL in namen
    except Exception:
        return False


def _vraag(prompt: str) -> str:
    body = json.dumps({"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "think": False,
                       "options": {"num_ctx": 16384, "temperature": 0.2}}).encode()
    req = urllib.request.Request(OLLAMA + "/api/generate", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        tekst = json.load(r)["response"]
    if "</think>" in tekst:
        tekst = tekst.split("</think>", 1)[1]
    return tekst.strip()


def _stukken(alineas, max_tekens=24000):
    """Deel het transcript op in stukken die in het geheugen van het model passen.
    Geeft (regels, starttijd, eindtijd) per stuk."""
    stuk, lengte, begin = [], 0, None
    for a in alineas:
        regel = f"[{tijd(a['start'])}] {a['tekst']}"
        if stuk and lengte + len(regel) > max_tekens:
            yield stuk, begin, eind
            stuk, lengte = [], 0
        if not stuk:
            begin = a["start"]
        stuk.append(regel)
        lengte += len(regel)
        eind = a["eind"]
    if stuk:
        yield stuk, begin, eind


OPBOUW_PROMPT = """Je krijgt {welk} van het transcript van een video, met tijden tussen [ ].
Dit stuk loopt van {van} tot {tot}.{vervolg}
Beschrijf in het Nederlands hoe dit stuk is opgebouwd: verdeel het in HOOFDonderdelen
(bijvoorbeeld intro, de belangrijkste onderwerpen, voorbeelden, afsluiting).
Regels:
- ongeveer {aantal} onderdelen; maak ze niet te klein
- de onderdelen sluiten op elkaar aan en dekken samen het hele stuk van {van} tot {tot}
- geef ALLEEN regels in exact dit formaat, één per onderdeel, niets anders:
mm:ss - mm:ss | korte titel | één zin wat er gebeurt

Transcript:
{tekst}"""

SAMENVATTING_PROMPT = """Hieronder staat informatie over een video{titel}.
Schrijf in het Nederlands een duidelijke samenvatting van 3 tot 6 zinnen over waar de video
over gaat en wat de belangrijkste punten zijn. Baseer je alleen op wat er echt gezegd wordt,
verzin niets. Is de video kort of bevat hij weinig inhoud, schrijf dan minder zinnen.
Geef alleen de samenvatting.

Opbouw:
{opbouw}
{transcript}"""


def opbouw_en_samenvatting(alineas, titel: str, voortgang=lambda *_: None):
    stukken = list(_stukken(alineas))
    regels = []
    n = len(stukken)
    for i, (stuk, van, tot) in enumerate(stukken, 1):
        voortgang(i / (n + 1), f"Opbouw bepalen (deel {i}/{n})...")
        minuten = (tot - van) / 60
        antwoord = _vraag(OPBOUW_PROMPT.format(
            welk="het hele transcript" if n == 1 else f"deel {i} van {n}",
            van=tijd(van), tot=tijd(tot),
            vervolg="" if i == 1 else (" Het is een vervolg: begin dus NIET met een inleiding,"
                                       " tenzij er echt een nieuwe inleiding is."),
            aantal=max(2, min(10, round(minuten / 2))), tekst="\n".join(stuk)))
        regels += [r.strip(" -*•") for r in antwoord.splitlines() if "|" in r]
    voortgang(len(stukken) / (len(stukken) + 1), "Samenvatting schrijven...")
    samenvatting = _vraag(SAMENVATTING_PROMPT.format(
        titel=f' met de titel "{titel}"' if titel else "", opbouw="\n".join(regels),
        transcript=("\nTranscript:\n" + "\n".join(stukken[0][0])) if n == 1 else ""))
    return regels, samenvatting


def opbouw_uit_pauzes(alineas, aantal=8):
    """Reserve als Ollama niet draait: deel de video op bij de langste pauzes."""
    if not alineas:
        return []
    grenzen = sorted(sorted(range(1, len(alineas)), key=lambda i: -alineas[i]["pauze_ervoor"])
                     [:aantal - 1])
    starts = [0] + grenzen
    regels = []
    for n, s in enumerate(starts):
        e = (starts[n + 1] if n + 1 < len(starts) else len(alineas)) - 1
        begin = " ".join(alineas[s]["tekst"].split()[:10])
        regels.append(f"{tijd(alineas[s]['start'])} - {tijd(alineas[e]['eind'])} | "
                      f"Deel {n + 1} | begint met: \"{begin}...\"")
    return regels


# ---------------------------------------------------------------- export ---
def tekstbestand(meta, taal, alineas, pauzes, opbouw, samenvatting, pauze_min, alinea_pauze,
                 opbouw_bron):
    woorden = sum(len(a["tekst"].split()) for a in alineas)
    spreektijd = sum(a["eind"] - a["start"] for a in alineas) or 1
    regels = [f"TITEL:   {meta.get('titel') or '(onbekend)'}"]
    if meta.get("kanaal"):
        regels.append(f"KANAAL:  {meta['kanaal']}")
    if meta.get("link"):
        regels.append(f"LINK:    {meta['link']}")
    regels += [f"DUUR:    {tijd(meta.get('duur') or (alineas[-1]['eind'] if alineas else 0))}",
               f"TAAL:    {taal}",
               f"GEMAAKT: {time.strftime('%d-%m-%Y %H:%M')}", ""]

    if samenvatting:
        regels += ["=" * 70, "SAMENVATTING", "=" * 70, samenvatting, ""]

    regels += ["=" * 70, f"OPBOUW VAN DE VIDEO ({opbouw_bron})", "=" * 70]
    regels += opbouw or ["(geen opbouw beschikbaar)"]
    regels.append("")

    regels += ["=" * 70, "TRANSCRIPT", "=" * 70,
               f"[pauze X.Xs] = stilte van minstens {pauze_min:.1f} seconde midden in een zin.",
               f"Nieuwe alinea met '--- pauze X.Xs ---' = stilte van {alinea_pauze:.1f} seconde of meer.",
               ""]
    hoofdstukken = list(meta.get("hoofdstukken") or [])
    for i, a in enumerate(alineas):
        while hoofdstukken and hoofdstukken[0][0] <= a["start"] + 0.5:
            start, naam = hoofdstukken.pop(0)
            regels += ["", f"## HOOFDSTUK: {naam}  ({tijd(start)})", ""]
        if i > 0 and a["pauze_ervoor"] >= alinea_pauze:
            regels.append(f"--- pauze {a['pauze_ervoor']:.1f}s ---")
        regels += [f"[{tijd(a['start'])}] {a['tekst']}", ""]

    langste = max((p[1] for p in pauzes), default=0)
    regels += ["=" * 70, "STATISTIEK", "=" * 70,
               f"Woorden:          {woorden}",
               f"Spreektempo:      {woorden / spreektijd * 60:.0f} woorden per minuut",
               f"Aantal pauzes:    {len(pauzes)} (vanaf {pauze_min:.1f}s)",
               f"Langste pauze:    {langste:.1f}s",
               f"Totale stilte:    {sum(p[1] for p in pauzes):.0f}s"]
    return "\n".join(regels) + "\n"


# ------------------------------------------------------------- alles samen ---
def video_naar_tekst(url: str | None, bestand: str | None, werkmap: Path, pauze_min=1.0,
                     alinea_pauze=2.5, met_ai=True, taal=None, voortgang=lambda *_: None):
    tmp = werkmap / "tmp"
    if bestand:
        pad, meta = bestand, {"titel": Path(bestand).stem, "hoofdstukken": []}
        verwijder = False
    else:
        voortgang(0.02, "Video ophalen van YouTube...")
        pad, meta = youtube_ophalen(url, tmp)
        verwijder = True

    voortgang(0.15, "Geluid voorbereiden...")
    audio = _audio_laden(pad)
    if verwijder:
        Path(pad).unlink(missing_ok=True)
    if not meta.get("duur"):
        meta["duur"] = len(audio) / SR

    voortgang(0.2, "Transcript maken (spraakherkenning)...")
    woorden, taal_gevonden = transcriberen(audio, taal)
    if not woorden:
        raise ValueError("Er is geen spraak gevonden in deze video.")
    alineas, pauzes = alinea_s(woorden, pauze_min, alinea_pauze,
                               [s for s, _ in meta.get("hoofdstukken") or []])

    samenvatting, opbouw = "", []
    if met_ai and ollama_beschikbaar():
        opbouw, samenvatting = opbouw_en_samenvatting(
            alineas, meta.get("titel", ""),
            lambda f, d: voortgang(0.6 + 0.38 * f, d))
        bron = f"gemaakt door AI ({OLLAMA_MODEL})"
    elif meta.get("hoofdstukken"):
        bron = "hoofdstukken van YouTube"
        h = meta["hoofdstukken"]
        eind = meta["duur"]
        opbouw = [f"{tijd(s)} - {tijd(h[i + 1][0] if i + 1 < len(h) else eind)} | {n}"
                  for i, (s, n) in enumerate(h)]
    else:
        bron = "ingedeeld op de langste pauzes (AI niet beschikbaar)"
        opbouw = opbouw_uit_pauzes(alineas)

    tekst = tekstbestand(meta, taal_gevonden, alineas, pauzes, opbouw, samenvatting,
                         pauze_min, alinea_pauze, bron)
    return tekst, meta
