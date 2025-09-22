"""Tests for the live music BPM analyzer."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

import pytest

from app import (
    AnalysisConfig,
    AudioAnalyzer,
    FileAudioSource,
    MicrophoneAudioSource,
    generate_click_track,
)


def write_wave(path: Path, audio: Sequence[float], sample_rate: int) -> None:
    """Persist ``audio`` as a mono WAV file for testing."""

    clipped = [max(-1.0, min(1.0, sample)) for sample in audio]
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for sample in clipped:
            int_value = int(round(sample * 32767))
            frames.extend(int_value.to_bytes(2, byteorder="little", signed=True))
        wf.writeframes(bytes(frames))


def make_frame_provider(
    segments: Sequence[Tuple[float, float]],
    sample_rate: int,
    chunk_size: int,
) -> Iterator[List[float]]:
    """Yield chunks that emulate microphone input for multiple tempo segments."""

    def generator() -> Iterable[List[float]]:
        for bpm, duration in segments:
            track = generate_click_track(bpm, duration, sample_rate)
            for start in range(0, len(track), chunk_size):
                chunk = list(track[start : start + chunk_size])
                if len(chunk) < chunk_size:
                    chunk.extend([0.0] * (chunk_size - len(chunk)))
                yield chunk

    return iter(generator())


def parse_log(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8")
    return [line for line in text.strip().splitlines() if line]


def test_file_source_detects_expected_bpm(tmp_path: Path) -> None:
    sample_rate = 44_100
    chunk_size = 4_096
    target_bpm = 120.0
    duration = 12.0
    audio = generate_click_track(target_bpm, duration, sample_rate)
    wav_path = tmp_path / "track.wav"
    write_wave(wav_path, audio, sample_rate)
    source = FileAudioSource(str(wav_path), sample_rate, chunk_size)
    log_path = tmp_path / "log.txt"
    config = AnalysisConfig(
        sample_rate=sample_rate,
        chunk_size=chunk_size,
        analysis_window=6.0,
        min_window=3.0,
        variation_threshold=2.0,
    )
    analyzer = AudioAnalyzer([source], str(log_path), config=config)
    analyzer.run()
    lines = parse_log(log_path)
    assert lines[0].startswith("EVENT"), "log should include header"
    baseline_line = next(line for line in lines if line.startswith("BASELINE"))
    baseline_bpm = float(baseline_line.split("\t")[2])
    assert baseline_bpm == pytest.approx(target_bpm, abs=2.0)


def test_microphone_variations_are_logged(tmp_path: Path) -> None:
    sample_rate = 44_100
    chunk_size = 2_048
    segments = [(100.0, 5.0), (120.0, 5.0)]
    provider = make_frame_provider(segments, sample_rate, chunk_size)
    microphone = MicrophoneAudioSource(
        sample_rate=sample_rate,
        chunk_size=chunk_size,
        frame_provider=provider,
    )
    log_path = tmp_path / "mic_log.txt"
    config = AnalysisConfig(
        sample_rate=sample_rate,
        chunk_size=chunk_size,
        analysis_window=5.0,
        min_window=3.0,
        variation_threshold=3.0,
    )
    analyzer = AudioAnalyzer([microphone], str(log_path), config=config)
    analyzer.run()
    lines = parse_log(log_path)
    variations = [line for line in lines if line.startswith("VARIATION")]
    assert variations, "expected at least one tempo variation entry"
    last_variation = variations[-1].split("\t")
    observed_bpm = float(last_variation[2])
    assert observed_bpm > 110.0
