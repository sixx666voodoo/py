# PY_SYNTH

Ein Python-GUI-Tool für folgende MIDI-Workflow-Funktionen:

1. MIDI-Datei laden & anzeigen
2. Engine wählen: FluidSynth oder BassMidi
3. SoundFont laden (.sf2/.sfz)
4. Vorschau mit Play/Stop + Volume
5. Export als WAV mit Statusanzeige

## Installation

1. Python 3.x installieren
2. Abhängigkeiten installieren:
   ```
   pip3 install -r requirements.txt
   ```

## Starten

```bash
python3 main.py
```

## Hinweise
- Unterstützt FluidSynth und BassMidi (sofern installiert)
- SoundFont-Formate: .sf2, .sfz
- MIDI- und WAV-Dateien werden lokal verarbeitet

---

## 📁 Projektstruktur

```
PY_SYNTH/
├── main.py                        # Haupt-GUI mit Synth-Management & Export
├── EngineSettingsPanel.py        # Dynamisches Einstellungsfenster für Engines
├── README.md                     # Projektbeschreibung und Setup-Hinweise
├── requirements.txt              # Abhängigkeiten (PyQt, fluidsynth, etc.)
│
├── resources/                    # (optional) App-Ressourcen
│   ├── icon.png                  # App-Icon für .app/.dmg Builds (Briefcase)
│   ├── libbass.dylib             # macOS-Bibliothek für BASS
│   └── libbassmidi.dylib         # macOS-Bibliothek für BASSMIDI
│
├── libs/                         # (optional) DLLs für Windows
│   ├── bass.dll
│   └── bassmidi.dll
│
├── tests/                        # (empfohlen) Unit-Tests mit pytest
│   └── test_engine_manager.py
│
└── pyproject.toml                # Konfiguration für Briefcase (.app Export)
```

---
