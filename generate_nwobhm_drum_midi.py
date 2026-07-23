"""Generate a 32-bar NWOBHM-style drum MIDI track at 145 BPM.

The output is a single-track Standard MIDI file (format 0) with:
- 4/4 time signature
- subtle deterministic timing offsets
- section-based velocity contouring
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

PPQ = 480
DEFAULT_BPM = 145

# General MIDI drum notes (channel 10 => index 9)
KICK = 36
SNARE = 38
CLOSED_HAT = 42
OPEN_HAT = 46
CRASH = 49
RIDE = 51
TOM_1 = 48
TOM_2 = 47
FLOOR_TOM = 43


class DrumMidiBuilder:
    def __init__(self, tempo_bpm: int = DEFAULT_BPM) -> None:
        self.tempo_bpm = tempo_bpm
        self.tempo_us = int(round(60_000_000 / tempo_bpm))
        self.events: list[tuple[int, str, int, int, int]] = []

    def add_note(self, start: int, duration: int, note: int, velocity: int, channel: int = 9) -> None:
        self.events.append((start, "on", channel, note, velocity))
        self.events.append((start + duration, "off", channel, note, 0))

    def add_hit(self, start: int, note: int, velocity: int, duration: int = 60, channel: int = 9) -> None:
        self.add_note(start, duration, note, velocity, channel)

    @staticmethod
    def bar_tick(bar_num: int) -> int:
        return (bar_num - 1) * 4 * PPQ

    @staticmethod
    def humanize_tick(tick: int, amount: int = 6) -> int:
        # deterministic subtle timing offsets
        return tick + ((tick // 60) % (2 * amount + 1) - amount)

    @staticmethod
    def vlq(value: int) -> bytes:
        data = [value & 0x7F]
        value >>= 7
        while value:
            data.append(0x80 | (value & 0x7F))
            value >>= 7
        return bytes(reversed(data))

    def section_velocity_multiplier(self, bar: int) -> float:
        if 1 <= bar <= 8:
            return 0.92
        if 9 <= bar <= 16:
            return 0.88
        if 17 <= bar <= 24:
            return 1.0
        if 25 <= bar <= 28:
            return 0.8
        return 1.05

    def velocity(self, base: int, bar: int, alt: int = 0) -> int:
        value = int(round((base + alt) * self.section_velocity_multiplier(bar)))
        return max(1, min(127, value))

    def beat_to_tick(self, bar: int, beat: int, sub: float = 0.0) -> int:
        return self.bar_tick(bar) + int(round(((beat - 1) + sub) * PPQ))

    def add_eighth_hats(self, bar: int, open_steps: set[int] | None = None, base_vel: int = 92) -> None:
        open_steps = open_steps or set()
        base_tick = self.bar_tick(bar)
        for i in range(8):
            tick = base_tick + i * (PPQ // 2)
            note = OPEN_HAT if i in open_steps else CLOSED_HAT
            velocity = self.velocity(base_vel, bar, alt=(4 if i in (0, 2, 4, 6) else -4))
            self.add_hit(self.humanize_tick(tick), note, velocity, duration=90 if note == OPEN_HAT else 75)

    def add_ride_eighths(self, bar: int, base_vel: int = 94) -> None:
        base_tick = self.bar_tick(bar)
        for i in range(8):
            tick = base_tick + i * (PPQ // 2)
            velocity = self.velocity(base_vel, bar, alt=(3 if i in (0, 2, 4, 6) else -3))
            self.add_hit(self.humanize_tick(tick, amount=4), RIDE, velocity, duration=80)

    def add_crash_downbeat(self, bar: int) -> None:
        self.add_hit(self.bar_tick(bar), CRASH, self.velocity(121, bar), duration=180)

    def add_kick(self, bar: int, positions: list[float], base: int = 114) -> None:
        for pos in positions:
            beat = int(math.floor(pos))
            frac = pos - beat
            tick = self.beat_to_tick(bar, beat, frac)
            alt = -6 if frac != 0 else 2
            self.add_hit(self.humanize_tick(tick, amount=5), KICK, self.velocity(base, bar, alt=alt), duration=70)

    def add_snare(self, bar: int, positions: tuple[float, ...] = (2, 4), base: int = 122) -> None:
        for pos in positions:
            beat = int(math.floor(pos))
            frac = pos - beat
            tick = self.beat_to_tick(bar, beat, frac)
            self.add_hit(self.humanize_tick(tick, amount=4), SNARE, self.velocity(base, bar, alt=2), duration=85)

    def add_fill(self, bar: int, pattern: list[tuple[float, int, int]]) -> None:
        for pos, note, vel in pattern:
            beat = int(math.floor(pos))
            frac = pos - beat
            tick = self.beat_to_tick(bar, beat, frac)
            self.add_hit(self.humanize_tick(tick, amount=3), note, self.velocity(vel, bar), duration=80)

    def arrange(self) -> None:
        for bar in range(1, 33):
            if bar in [17, 18, 19, 20, 21, 22, 23, 24, 29, 30, 31, 32]:
                if bar in [17, 21, 29]:
                    self.add_crash_downbeat(bar)
                if bar != 32:
                    self.add_ride_eighths(bar)
                else:
                    base_tick = self.bar_tick(bar)
                    for i in range(6):
                        tick = base_tick + i * (PPQ // 2)
                        vel = self.velocity(94, bar, alt=(3 if i % 2 == 0 else -3))
                        self.add_hit(self.humanize_tick(tick, 4), RIDE, vel, duration=80)
            else:
                open_steps: set[int] = set()
                if bar in [7, 15, 27]:
                    open_steps.add(7)
                self.add_eighth_hats(bar, open_steps=open_steps)
                if bar in [1, 5]:
                    self.add_crash_downbeat(bar)

            if bar == 1:
                self.add_kick(bar, [1, 1.5, 3, 3.5]); self.add_snare(bar)
            elif bar == 2:
                self.add_kick(bar, [1, 2.5, 3, 3.5]); self.add_snare(bar)
            elif bar == 3:
                self.add_kick(bar, [1, 1.5, 3, 4.5]); self.add_snare(bar)
            elif bar == 4:
                self.add_kick(bar, [1, 3]); self.add_snare(bar)
                self.add_fill(bar, [(4.25, TOM_1, 112), (4.5, TOM_2, 114), (4.75, FLOOR_TOM, 118)])
            elif bar == 5:
                self.add_kick(bar, [1, 1.5, 2.75, 3, 3.5]); self.add_snare(bar)
            elif bar == 6:
                self.add_kick(bar, [1, 2.5, 3, 4.75]); self.add_snare(bar)
            elif bar == 7:
                self.add_kick(bar, [1, 1.5, 3, 3.5]); self.add_snare(bar)
            elif bar == 8:
                self.add_kick(bar, [1, 3]); self.add_snare(bar)
                self.add_fill(bar, [(4.25, TOM_1, 112), (4.5, TOM_2, 114), (4.75, FLOOR_TOM, 118)])
            elif bar == 9:
                self.add_kick(bar, [1, 3], base=110); self.add_snare(bar, base=120)
            elif bar == 10:
                self.add_kick(bar, [1, 2.5, 3], base=110); self.add_snare(bar, base=120)
            elif bar == 11:
                self.add_kick(bar, [1, 1.5, 3], base=111); self.add_snare(bar, base=120)
            elif bar == 12:
                self.add_kick(bar, [1, 3, 4.5], base=111); self.add_snare(bar, base=120)
            elif bar == 13:
                self.add_kick(bar, [1, 1.5, 3, 3.5], base=112); self.add_snare(bar, base=121)
            elif bar == 14:
                self.add_kick(bar, [1, 2.5, 3], base=112); self.add_snare(bar, base=121)
            elif bar == 15:
                self.add_kick(bar, [1, 1.5, 3, 4.5], base=112); self.add_snare(bar, base=121)
            elif bar == 16:
                self.add_kick(bar, [1, 3], base=111); self.add_snare(bar, base=121)
                self.add_fill(bar, [(4.0, SNARE, 120), (4.5, TOM_1, 114), (4.75, FLOOR_TOM, 118)])
            elif bar == 17:
                self.add_kick(bar, [1, 1.5, 3, 3.5], base=116); self.add_snare(bar, base=123)
            elif bar == 18:
                self.add_kick(bar, [1, 2.5, 3, 4.5], base=116); self.add_snare(bar, base=123)
            elif bar == 19:
                self.add_kick(bar, [1, 1.5, 2.75, 3, 3.5], base=117); self.add_snare(bar, base=124)
            elif bar == 20:
                self.add_kick(bar, [1, 3, 3.5], base=116); self.add_snare(bar, base=123)
            elif bar == 21:
                self.add_kick(bar, [1, 1.5, 3, 3.5], base=117); self.add_snare(bar, base=124)
            elif bar == 22:
                self.add_kick(bar, [1, 2.5, 3, 4.5], base=117); self.add_snare(bar, base=124)
            elif bar == 23:
                self.add_kick(bar, [1, 1.5, 3, 3.5, 4.5], base=118); self.add_snare(bar, base=124)
            elif bar == 24:
                self.add_kick(bar, [1, 3], base=116); self.add_snare(bar, base=124)
                self.add_fill(bar, [(4.25, TOM_1, 114), (4.5, TOM_2, 116), (4.75, FLOOR_TOM, 120)])
            elif bar == 25:
                self.add_kick(bar, [1, 3], base=108); self.add_snare(bar, base=119)
            elif bar == 26:
                self.add_kick(bar, [1, 2.5, 3], base=108); self.add_snare(bar, base=119)
                self.add_fill(bar, [(4.75, FLOOR_TOM, 112)])
            elif bar == 27:
                self.add_kick(bar, [1, 1.5, 3], base=109); self.add_snare(bar, base=120)
            elif bar == 28:
                self.add_kick(bar, [1, 3], base=109); self.add_snare(bar, base=120)
                self.add_fill(bar, [(4.5, TOM_1, 112), (4.75, FLOOR_TOM, 116)])
            elif bar == 29:
                self.add_kick(bar, [1, 1.5, 3, 3.5], base=119); self.add_snare(bar, base=125)
            elif bar == 30:
                self.add_kick(bar, [1, 2.5, 3, 4.5], base=119); self.add_snare(bar, base=125)
            elif bar == 31:
                self.add_kick(bar, [1, 1.5, 2.75, 3, 3.5, 4.5], base=120); self.add_snare(bar, base=126)
            elif bar == 32:
                self.add_kick(bar, [1, 3], base=120); self.add_snare(bar, base=126)
                self.add_fill(bar, [(4.0, SNARE, 123), (4.25, TOM_1, 116), (4.5, TOM_2, 118), (4.75, FLOOR_TOM, 122)])

    def build(self) -> bytes:
        self.events.sort(key=lambda event: (event[0], 0 if event[1] == "off" else 1))

        def meta_event(delta: int, meta_type: int, data: bytes) -> bytes:
            return self.vlq(delta) + bytes([0xFF, meta_type]) + self.vlq(len(data)) + data

        def midi_event(delta: int, status: int, data: list[int]) -> bytes:
            return self.vlq(delta) + bytes([status]) + bytes(data)

        track = bytearray()
        track += meta_event(0, 0x03, b"NWOBHM 145 BPM Drum Track")
        track += meta_event(0, 0x51, self.tempo_us.to_bytes(3, "big"))
        track += meta_event(0, 0x58, bytes([4, 2, 24, 8]))
        track += meta_event(0, 0x59, bytes([0, 0]))

        last_tick = 0
        for tick, kind, channel, note, velocity in self.events:
            delta = max(0, int(tick - last_tick))
            status = (0x90 if kind == "on" else 0x80) | channel
            track += midi_event(delta, status, [note, velocity])
            last_tick = int(tick)

        track += meta_event(PPQ, 0x2F, b"")

        header = bytearray()
        header += b"MThd"
        header += struct.pack(">IHHH", 6, 0, 1, PPQ)

        return bytes(header) + b"MTrk" + struct.pack(">I", len(track)) + track


def generate(output_path: Path, bpm: int = DEFAULT_BPM) -> tuple[Path, int]:
    builder = DrumMidiBuilder(tempo_bpm=bpm)
    builder.arrange()
    midi_data = builder.build()
    output_path.write_bytes(midi_data)
    return output_path, len(midi_data)


if __name__ == "__main__":
    out, size = generate(Path("nwobhm_145bpm_drum_track.mid"), bpm=145)
    print(f"Wrote {out} ({size} bytes)")
