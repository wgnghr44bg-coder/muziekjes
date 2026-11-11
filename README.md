# 🎵 Muziek Kloon

Lokale web-app (Gradio) om achtergrondmuziek uit een nummer of video te halen en na te maken.
Draait op een eigen pc met NVIDIA-videokaart en is via een openbare link ook op de telefoon te gebruiken.

## Functies

| Tabblad | Wat het doet |
|---|---|
| **1. Achtergrond eruit halen** | Haalt zang/stem weg met Demucs (`htdemucs`). Geeft de achtergrond plus losse sporen (drums, bas, overig, zang). Accepteert audio én video (mp4, mov, ...). |
| **2. Nieuwe achtergrond maken** | MusicGen-Melody maakt nieuwe muziek (max. 30 sec) die de melodie en stijl van een voorbeeld volgt. |
| **MP4 → MP3** | Zet een video om naar mp3 (192 kbps). |

## Starten

1. Vereist: Windows, Python 3.12, [ffmpeg](https://www.gyan.dev/ffmpeg/builds/) in PATH, NVIDIA-videokaart (8 GB werkt).
2. Dubbelklik **`start.bat`**. De eerste keer installeert hij alles in `.venv` (enkele GB).
3. De app opent in de browser. Het wachtwoord staat in `wachtwoord.txt` (wordt de eerste keer
   automatisch aangemaakt en staat **niet** op GitHub).

## Openbare link / GitHub Pages

Bij elke start maakt Gradio een tijdelijke openbare link (`*.gradio.live`). `app.py` zet die link
automatisch in `docs/index.html` en pusht naar GitHub. GitHub Pages (map `/docs`) stuurt bezoekers van
het vaste adres door naar de app. De link werkt alleen zolang de app op de pc draait.

## Bestanden

- `app.py`: de hele app (verwerking + interface + login + link publiceren)
- `start.bat`: installeren en starten
- `requirements.txt`: Python-pakketten (PyTorch wordt apart geïnstalleerd met CUDA 12.6)
- `docs/sjabloon.html`: doorstuurpagina, `{{URL}}` wordt vervangen door de actuele link
- `uitvoer/`: resultaten (niet op GitHub)

## Aanpassen

- Login zit in de app zelf (`inloggen` / `alleen_ingelogd` in `app.py`), zonder cookies, zodat het ook op iPhone werkt.
- Stap 1 gebruikt bewust `htdemucs` zonder nabewerking: dat klonk bij het luisteren het beste.
  `htdemucs_ft` liet meer stem over; BS-RoFormer + ruisfilter haalde meer stem weg maar klonk minder goed.
