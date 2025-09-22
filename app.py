"""Live music analysis tool for tempo tracking.

The module exposes a command line interface and a reusable API that can analyse
up to two audio sources (file and microphone) in real time. It uses a lightweight
machine-learning tempo regressor that is trained on synthetic percussion
patterns generated at start-up. No third-party dependencies are required.
"""

from __future__ import annotations

import argparse
import math
import os
import queue
import sys
import wave
from dataclasses import dataclass
from statistics import mean, median
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple


@dataclass
class AnalysisConfig:
    """Configuration options for :class:`AudioAnalyzer`."""

    sample_rate: int = 44_100
    chunk_size: int = 4_096
    analysis_window: float = 8.0
    min_window: float = 3.5
    variation_threshold: float = 2.0


class AudioSource:
    """Base class for audio sources feeding the analyzer."""

    def __init__(self, sample_rate: int, chunk_size: int) -> None:
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

    def read_chunk(self) -> Optional[List[float]]:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - default cleanup
        return None


class FileAudioSource(AudioSource):
    """Audio source reading samples from a waveform file."""

    def __init__(self, path: str, sample_rate: int, chunk_size: int) -> None:
        super().__init__(sample_rate=sample_rate, chunk_size=chunk_size)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self._data = self._load_file(path)
        self._position = 0

    def _load_file(self, path: str) -> List[float]:
        with wave.open(path, "rb") as wf:
            n_frames = wf.getnframes()
            sample_width = wf.getsampwidth()
            dtype_map = {1: "b", 2: "h", 4: "i"}
            typecode = dtype_map.get(sample_width)
            if typecode is None:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            raw = wf.readframes(n_frames)
            channels = wf.getnchannels()
            data = list(_bytes_to_samples(raw, typecode, channels))
            original_rate = wf.getframerate()
        if original_rate != self.sample_rate:
            data = resample_audio(data, original_rate, self.sample_rate)
        return data

    def read_chunk(self) -> Optional[List[float]]:
        if self._position >= len(self._data):
            return None
        end = min(self._position + self.chunk_size, len(self._data))
        chunk = self._data[self._position:end]
        self._position = end
        if len(chunk) < self.chunk_size:
            chunk = chunk + [0.0] * (self.chunk_size - len(chunk))
        return chunk


class MicrophoneAudioSource(AudioSource):
    """Audio source capturing data from a microphone.

    The optional ``frame_provider`` argument accepts an iterator yielding lists
    of floats, enabling deterministic unit tests without requiring audio
    hardware.
    """

    def __init__(
        self,
        sample_rate: int,
        chunk_size: int,
        frame_provider: Optional[Iterator[List[float]]] = None,
    ) -> None:
        super().__init__(sample_rate=sample_rate, chunk_size=chunk_size)
        self._frame_provider = frame_provider
        self._queue: "queue.Queue[List[float]]" = queue.Queue()
        self._stream = None
        self._sounddevice = None
        if frame_provider is None:
            try:
                import sounddevice  # type: ignore
            except ImportError as exc:  # pragma: no cover - hardware dependent
                raise RuntimeError(
                    "sounddevice is required for microphone input"
                ) from exc
            self._sounddevice = sounddevice

    def _ensure_stream(self) -> None:  # pragma: no cover - depends on hardware
        if self._stream is None and self._sounddevice is not None:
            self._stream = self._sounddevice.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                blocksize=self.chunk_size,
                callback=self._callback,
            )
            self._stream.start()

    def _callback(self, indata, frames, time_info, status) -> None:  # pragma: no cover
        if status:
            print(f"Microphone status: {status}", file=sys.stderr)
        mono: List[float] = []
        for frame in indata:
            if isinstance(frame, (list, tuple)):
                mono.append(float(sum(frame) / len(frame)))
            else:  # Fallback when the backend provides a scalar per frame
                mono.append(float(frame))
        self._queue.put(mono)

    def read_chunk(self) -> Optional[List[float]]:
        if self._frame_provider is not None:
            try:
                return next(self._frame_provider)
            except StopIteration:
                return None
        self._ensure_stream()
        try:
            return self._queue.get(timeout=1.0)
        except queue.Empty:
            return [0.0] * self.chunk_size

    def close(self) -> None:  # pragma: no cover - depends on hardware
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class BpmLogWriter:
    """Utility class writing tempo measurements to disk."""

    def __init__(self, path: str, variation_threshold: float) -> None:
        self._variation_threshold = variation_threshold
        self._file = open(path, "w", encoding="utf-8")
        self._file.write(
            "EVENT\ttimestamp_seconds\tobserved_bpm\treference_bpm\tdelta_bpm\n"
        )

    def log_baseline(self, bpm: float) -> None:
        self._file.write(f"BASELINE\t0.00\t{bpm:.2f}\t{bpm:.2f}\t+0.00\n")
        self._file.flush()

    def log_measurement(self, timestamp: float, bpm: float, reference: float) -> None:
        delta = bpm - reference
        label = "VARIATION" if abs(delta) >= self._variation_threshold else "MEASUREMENT"
        self._file.write(
            f"{label}\t{timestamp:.2f}\t{bpm:.2f}\t{reference:.2f}\t{delta:+.2f}\n"
        )
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class TempoRegressor:
    """Machine learning model predicting tempo from audio buffers."""

    def __init__(
        self,
        sample_rate: int,
        min_bpm: float = 60.0,
        max_bpm: float = 200.0,
        frame_size: int = 2_048,
        hop_size: int = 512,
    ) -> None:
        self.sample_rate = sample_rate
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.frame_size = frame_size
        self.hop_size = hop_size
        self._weights = self._train_model()

    def predict(self, audio: List[float]) -> Optional[float]:
        if len(audio) < self.frame_size:
            return None
        envelope = self._onset_envelope(audio)
        feature_data = self._extract_features(envelope)
        if feature_data is None:
            return None
        base_bpm, features = feature_data
        augmented = features + [1.0]
        adjustment = dot_product(augmented, self._weights)
        bpm = base_bpm + adjustment
        return max(self.min_bpm, min(self.max_bpm, bpm))

    def _train_model(self) -> List[float]:
        samples = 40
        bpms = [self.min_bpm + i * (self.max_bpm - self.min_bpm) / (samples - 1) for i in range(samples)]
        feature_rows: List[List[float]] = []
        targets: List[float] = []
        for bpm in bpms:
            audio = generate_click_track(bpm, 8.0, self.sample_rate)
            envelope = self._onset_envelope(audio)
            feature_data = self._extract_features(envelope)
            if feature_data is None:
                continue
            base_bpm, features = feature_data
            feature_rows.append(features)
            targets.append(bpm - base_bpm)
        if not feature_rows:
            raise RuntimeError("Unable to train tempo regressor: empty feature set")
        # Build normal equations for linear regression with bias term.
        augmented = [row + [1.0] for row in feature_rows]
        cols = len(augmented[0])
        normal_matrix = [[0.0 for _ in range(cols)] for _ in range(cols)]
        normal_vector = [0.0 for _ in range(cols)]
        for row, target in zip(augmented, targets):
            for i in range(cols):
                normal_vector[i] += row[i] * target
                for j in range(cols):
                    normal_matrix[i][j] += row[i] * row[j]
        weights = solve_linear_system(normal_matrix, normal_vector)
        return weights

    def _onset_envelope(self, audio: List[float]) -> List[float]:
        frame_size = self.frame_size
        hop = self.hop_size
        if len(audio) < frame_size:
            return []
        prev_energy = 0.0
        envelope: List[float] = []
        for start in range(0, len(audio) - frame_size + 1, hop):
            frame = audio[start : start + frame_size]
            energy = sum(abs(x) for x in frame)
            envelope.append(max(0.0, energy - prev_energy))
            prev_energy = energy
        normalize(envelope)
        return envelope

    def _extract_features(self, envelope: List[float]) -> Optional[Tuple[float, List[float]]]:
        if not envelope:
            return None
        frame_rate = self.sample_rate / self.hop_size
        peaks = detect_peaks(envelope)
        autocorr = autocorrelation(envelope)
        if not autocorr:
            return None
        min_lag = max(1, int(round(frame_rate * 60.0 / self.max_bpm)))
        max_lag = min(len(autocorr) - 1, int(round(frame_rate * 60.0 / self.min_bpm)))
        if max_lag <= min_lag:
            return None
        best_index = max_range_index(autocorr, min_lag, max_lag)
        lag_frames = best_index
        if len(peaks) >= 2:
            intervals = [peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)]
            lag_frames = max(1, int(round(median(intervals))))
        best_value = autocorr[min(lag_frames, len(autocorr) - 1)]
        secondary_index, secondary_value = secondary_peak(autocorr, best_index, min_lag, max_lag)
        lag_seconds = lag_frames / frame_rate
        secondary_seconds = secondary_index / frame_rate
        base_bpm = 60.0 / lag_seconds if lag_seconds > 0 else 0.0
        secondary_ratio = (
            (secondary_value / best_value) if best_value > 0 else 0.0
        )
        features = [
            best_value,
            secondary_value,
            secondary_ratio,
            secondary_seconds,
            mean(envelope) if envelope else 0.0,
        ]
        return base_bpm, features


class AudioAnalyzer:
    """Coordinate audio streaming and tempo analysis."""

    def __init__(
        self,
        sources: Sequence[AudioSource],
        log_path: str,
        config: Optional[AnalysisConfig] = None,
    ) -> None:
        if not sources:
            raise ValueError("At least one audio source must be provided")
        self.sources = list(sources)
        self.config = config or AnalysisConfig()
        self.log_writer = BpmLogWriter(log_path, self.config.variation_threshold)
        self._tempo_model = TempoRegressor(self.config.sample_rate)
        self._buffer: List[float] = []
        self._processed_samples = 0
        self._baseline: Optional[float] = None
        self._reference_bpm: Optional[float] = None
        self._active_sources = [True for _ in self.sources]

    def run(self, max_duration: Optional[float] = None) -> None:
        try:
            while True:
                chunk = self._mix_sources()
                if chunk is None:
                    break
                self._processed_samples += len(chunk)
                self._append_to_buffer(chunk)
                self._maybe_log_measurement()
                if (
                    max_duration is not None
                    and self._processed_samples / self.config.sample_rate >= max_duration
                ):
                    break
        finally:
            for source in self.sources:
                source.close()
            self.log_writer.close()

    def _mix_sources(self) -> Optional[List[float]]:
        chunks: List[Optional[List[float]]] = []
        max_length = 0
        any_active = False
        for index, source in enumerate(self.sources):
            if not self._active_sources[index]:
                chunks.append(None)
                continue
            chunk = source.read_chunk()
            if chunk is None:
                self._active_sources[index] = False
                chunks.append(None)
                continue
            if not chunk:
                chunks.append(None)
                continue
            any_active = True
            max_length = max(max_length, len(chunk))
            chunks.append(chunk)
        if not any_active:
            return None
        mix = [0.0] * max_length
        for chunk in chunks:
            if chunk is None:
                continue
            padded = chunk + [0.0] * (max_length - len(chunk))
            for i, value in enumerate(padded):
                mix[i] += value
        return mix

    def _append_to_buffer(self, chunk: List[float]) -> None:
        self._buffer.extend(chunk)
        max_samples = int(self.config.analysis_window * self.config.sample_rate)
        if len(self._buffer) > max_samples:
            self._buffer = self._buffer[-max_samples:]

    def _maybe_log_measurement(self) -> None:
        min_samples = int(self.config.min_window * self.config.sample_rate)
        if len(self._buffer) < min_samples:
            return
        bpm = self._tempo_model.predict(self._buffer)
        if bpm is None:
            return
        if self._baseline is None:
            self._baseline = bpm
            self._reference_bpm = bpm
            self.log_writer.log_baseline(bpm)
            return
        assert self._reference_bpm is not None
        timestamp = self._processed_samples / self.config.sample_rate
        self.log_writer.log_measurement(timestamp, bpm, self._reference_bpm)
        smoothing = 0.9
        self._reference_bpm = smoothing * self._reference_bpm + (1.0 - smoothing) * bpm


def _bytes_to_samples(raw: bytes, typecode: str, channels: int) -> Iterable[float]:
    """Convert raw wave bytes into normalised mono samples."""

    if typecode == "b":
        max_abs = 127.0
        sample_size = 1
    elif typecode == "h":
        max_abs = 32767.0
        sample_size = 2
    elif typecode == "i":
        max_abs = 2_147_483_647.0
        sample_size = 4
    else:  # pragma: no cover - defensive programming
        raise ValueError(f"Unsupported typecode: {typecode}")
    values: List[float] = []
    step = channels * sample_size
    for idx in range(0, len(raw), step):
        frame_total = 0.0
        for ch in range(channels):
            start = idx + ch * sample_size
            sample_bytes = raw[start : start + sample_size]
            sample_int = int.from_bytes(sample_bytes, byteorder="little", signed=True)
            frame_total += sample_int
        mono_value = frame_total / max(1, channels)
        values.append(mono_value / max_abs)
    return values


def normalize(values: List[float]) -> None:
    max_value = max((abs(x) for x in values), default=0.0)
    if max_value > 0:
        for i, value in enumerate(values):
            values[i] = value / max_value


def dot_product(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def solve_linear_system(matrix: List[List[float]], vector: List[float]) -> List[float]:
    size = len(matrix)
    augmented = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for i in range(size):
        pivot = augmented[i][i]
        if abs(pivot) < 1e-9:
            for j in range(i + 1, size):
                if abs(augmented[j][i]) > 1e-9:
                    augmented[i], augmented[j] = augmented[j], augmented[i]
                    pivot = augmented[i][i]
                    break
            else:
                pivot = 1e-9
        factor = 1.0 / pivot
        for k in range(i, size + 1):
            augmented[i][k] *= factor
        for j in range(i + 1, size):
            scale = augmented[j][i]
            if scale == 0.0:
                continue
            for k in range(i, size + 1):
                augmented[j][k] -= scale * augmented[i][k]
    solution = [0.0 for _ in range(size)]
    for i in range(size - 1, -1, -1):
        value = augmented[i][size]
        for k in range(i + 1, size):
            value -= augmented[i][k] * solution[k]
        solution[i] = value
    return solution


def autocorrelation(values: List[float]) -> List[float]:
    n = len(values)
    if n == 0:
        return []
    result = [0.0 for _ in range(n)]
    for lag in range(n):
        total = 0.0
        for i in range(n - lag):
            total += values[i] * values[i + lag]
        result[lag] = total
    normalize(result)
    return result


def max_range_index(values: List[float], start: int, end: int) -> int:
    best_index = start
    best_value = values[start]
    for i in range(start + 1, end + 1):
        if values[i] > best_value:
            best_value = values[i]
            best_index = i
    return best_index


def secondary_peak(
    values: List[float],
    primary_index: int,
    start: int,
    end: int,
) -> Tuple[int, float]:
    span = 2
    secondary_index = primary_index
    secondary_value = 0.0
    for i in range(start, end + 1):
        if abs(i - primary_index) <= span:
            continue
        if values[i] > secondary_value:
            secondary_value = values[i]
            secondary_index = i
    return secondary_index, secondary_value


def detect_peaks(envelope: List[float]) -> List[int]:
    if not envelope:
        return []
    maximum = max(envelope)
    if maximum <= 0:
        return []
    threshold = maximum * 0.3
    peaks: List[int] = []
    for i in range(1, len(envelope) - 1):
        if envelope[i] < threshold:
            continue
        if envelope[i] >= envelope[i - 1] and envelope[i] >= envelope[i + 1]:
            peaks.append(i)
    return peaks


def resample_audio(audio: List[float], original_rate: int, target_rate: int) -> List[float]:
    if original_rate == target_rate or not audio:
        return list(audio)
    duration = len(audio) / float(original_rate)
    target_length = max(1, int(round(duration * target_rate)))
    result = []
    for i in range(target_length):
        position = i / target_rate * original_rate
        left = int(math.floor(position))
        right = min(len(audio) - 1, left + 1)
        if right == left:
            result.append(audio[left])
        else:
            frac = position - left
            result.append(audio[left] * (1 - frac) + audio[right] * frac)
    return result


def generate_click_track(bpm: float, duration: float, sample_rate: int) -> List[float]:
    total_samples = int(duration * sample_rate)
    if total_samples <= 0 or bpm <= 0:
        return [0.0] * max(0, total_samples)
    signal = [0.0 for _ in range(total_samples)]
    interval = max(1, int(round(sample_rate * 60.0 / bpm)))
    pulse_length = max(1, int(0.01 * sample_rate))
    for start in range(0, total_samples, interval):
        for offset in range(pulse_length):
            idx = start + offset
            if idx >= total_samples:
                break
            envelope = 1.0 - offset / pulse_length
            signal[idx] += envelope
    normalize(signal)
    return signal


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live music BPM analyzer")
    parser.add_argument("--file", dest="file_path", help="Path to an audio file")
    parser.add_argument(
        "--microphone",
        dest="use_microphone",
        action="store_true",
        help="Enable live microphone input",
    )
    parser.add_argument(
        "--log",
        dest="log_path",
        default="bpm_log.txt",
        help="Destination log file",
    )
    parser.add_argument(
        "--duration",
        dest="duration",
        type=float,
        default=None,
        help="Optional duration in seconds to limit the analysis run",
    )
    parser.add_argument(
        "--sample-rate",
        dest="sample_rate",
        type=int,
        default=AnalysisConfig.sample_rate,
        help="Target sample rate for processing",
    )
    parser.add_argument(
        "--chunk-size",
        dest="chunk_size",
        type=int,
        default=AnalysisConfig.chunk_size,
        help="Frame size for streaming audio",
    )
    return parser.parse_args(argv)


def create_sources(args: argparse.Namespace) -> List[AudioSource]:
    sources: List[AudioSource] = []
    sample_rate = args.sample_rate
    chunk_size = args.chunk_size
    if args.file_path:
        sources.append(FileAudioSource(args.file_path, sample_rate, chunk_size))
    if args.use_microphone:
        sources.append(MicrophoneAudioSource(sample_rate, chunk_size))
    if not sources:
        raise SystemExit("No audio sources selected. Specify --file and/or --microphone.")
    return sources


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    sources = create_sources(args)
    config = AnalysisConfig(sample_rate=args.sample_rate, chunk_size=args.chunk_size)
    analyzer = AudioAnalyzer(sources, log_path=args.log_path, config=config)
    analyzer.run(max_duration=args.duration)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
