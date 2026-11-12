# Muziekjes – afspraken voor Claude

Zie README.md voor wat de app doet en hoe hij werkt. De gebruiker spreekt Nederlands.

## Keuzes van de gebruiker (niet zonder overleg terugdraaien)
- Stap 1 gebruikt bewust `htdemucs` **zonder** nabewerking (geen ruisfilter, geen normaliseren).
  De gebruiker vond dit het beste klinken; `htdemucs_ft` en BS-RoFormer + ruisfilter klonken slechter.
- Hosting via de eigen pc (Gradio share-link + GitHub Pages-doorstuurpagina), niet via betaalde hosting.
- Login zit in de app zelf (`inloggen` / `alleen_ingelogd`), niet via Gradio `auth=`:
  die cookie-login gaf op iPhone "401 Not authenticated" bij uploaden.
- De gebruiker gebruikt de app vooral op de iPhone, en uploadt vaak schermopnames (mp4).

- Ingelogd blijven: de site (docs/sjabloon.html) en de app rekenen dezelfde sleutel uit
  (sha256 van "muziekjes:" + wachtwoord) en geven die door als `?t=`. Wijzig je dat, pas dan beide aan.
- Video → tekst en Naar MP3 werken met YouTube én TikTok. TikTok heeft `yt-dlp[curl-cffi]` nodig
  (anders blokkeert TikTok). Opbouw/samenvatting via lokale Ollama `qwen3:14b`.
- Op de iPhone kan een website niet op de achtergrond opnemen; pc-opname (loopback) wel.

## Praktisch
- Bestanden schrijven/lezen vanuit Windows PowerShell 5.1: altijd UTF-8 opgeven (anders raken é, → en emoji stuk).
- `wachtwoord.txt` staat nooit op GitHub. Zet hem nooit in code, commits of README.
- Na codewijzigingen: commit + push naar `main`, sluit oude app-instanties en start `start.bat`
  opnieuw. De app pusht zelf de nieuwe link naar `docs/index.html`.
- De GPU (RTX 2080 SUPER, 8 GB) wordt gedeeld met ComfyUI, Ollama en Omni voice (vox-lux).
  Meerdere app-instanties tegelijk maken alles heel traag.
- GitHub CLI staat niet in PATH: `%LOCALAPPDATA%\gh-cli\bin\gh.exe`.
