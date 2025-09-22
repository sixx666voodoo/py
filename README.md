# Live Music BPM Analyzer

This repository provides a command-line tool that performs live tempo analysis
on one or two audio sources. The analyzer can combine microphone input with a
file-based track, estimate the baseline tempo using a lightweight machine
learning model, and continuously log variations in speed.

## Features

- Stream audio from a microphone, an audio file, or both at the same time.
- Predict tempo (BPM) with a model trained on synthetic percussion patterns.
- Track live tempo variations and write detailed entries to a log file.
- Includes reusable Python classes for integration into other projects.

## Installation

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

Run the analyzer from the command line:

```bash
python app.py --file path/to/song.wav --microphone --log bpm_log.txt
```

Key options:

- `--file`: optional path to a wave file that should be analysed.
- `--microphone`: enables microphone capture (requires the `sounddevice`
  package and audio hardware).
- `--log`: location of the tempo log (defaults to `bpm_log.txt`).
- `--duration`: optional limit in seconds; the analyzer stops when the duration
  is exceeded.
- `--sample-rate` and `--chunk-size`: configure the processing parameters.

The generated log contains a header row and records the event type, timestamp,
observed BPM, reference BPM (smoothed baseline) and the difference between the
latest measurement and the baseline. Lines marked as `VARIATION` exceed the
configured deviation threshold.

## Testing

Execute the automated tests with:

```bash
pytest
```
