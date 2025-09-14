DeepFilterNet2 Live (macOS)

En enkel macOS‑desktopapp i Python (Tkinter) som tar rått ljud från din mikrofon, kör DeepFilterNet2 i nära realtid, och spelar upp resultatet i högtalarna. Den har en Start/Stop‑knapp och en slider för hur mycket separation (attenuering) som tillåts.

Förutsättningar
- macOS med Xcode CLT (för Rust)
- Rust toolchain (rustup)
- Python 3.9+ med `pip`

Bygg libDF C‑API (dylib)
```bash
# I repo-roten
cargo build -p deep_filter --release --features "capi,default-model"
```

Vad byggs?
- macOS dynlib: `target/release/libdf.dylib` (Cargo‑workspace; standardnamn)
- Appen letar efter `libdf.dylib` eller `libdeepfilter.dylib` i `target/{release,debug}`.

Installera Python‑beroenden
```bash
pip install sounddevice numpy
```

Starta appen
```bash
# Rekommenderat: använd DF3‑modellen (C‑API stöder ej DF2 i denna version)
export DF_MODEL_TAR=models/DeepFilterNet3_onnx.tar.gz
python macos_app/app.py
```

Användning i appen
- Välj din mikrofon och dina högtalare i dropdownlistorna.
- Klicka Start för att börja strömma ljudet genom DeepFilterNet2.
- Justera “Separationsnivå (dB)” för mindre/mer suppression:
  - 0 dB ≈ nästan ingen suppressionsblandning (mer originalljud kvar)
  - 40 dB ≈ stark suppression (minimerar bakgrundsljud)
- “Latency/RT factor” visar medel processeringstid per block och hur nära realtid du är:
  - RT factor < 1.00 betyder att processeringen håller jämna steg med realtid.

Felsökning
- Enhetsfel: Om ett ljudkort inte stödjer 48 kHz/mono visas fel. Välj annan enhet.
- Brus/artefakter: Sänk separationsnivån (färre dB) eller undvik överstyrning på mic.
- Tillåtelse: macOS kan kräva mikrofonåtkomst vid första körning (System Settings → Privacy & Security → Microphone).
- Dylib saknas: Säkerställ att `target/release/libdf.dylib` finns efter build. Alternativt starta med env:
  - `DEEPFILTER_DYLIB=/full/path/libdf.dylib python macos_app/app.py`
- DF2‑modell kraschar: Den här C‑API‑versionen stöder inte DeepFilterNet2‑modeller. Använd DF3 (`models/DeepFilterNet3_onnx.tar.gz`) eller be mig aktivera en ren Python‑väg för DF2‑streaming.

Tips
- Slider: 0 dB = nästan ingen suppression, 40 dB = stark suppression.
- Appen antar 48 kHz, mono. Sätt din input‑device till 48 kHz för bäst resultat.
- Om du får PortAudio‑fel: välj rätt input/output enheter via `sounddevice.default.device` eller `sd.Stream(device=(in_id, out_id), ...)` i koden.
