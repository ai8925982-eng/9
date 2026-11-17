"""Rule-based + stochastic composer: turns a genre profile into MIDI-like tracks."""

from __future__ import annotations

import random

import numpy as np
from dataclasses import dataclass, field

from .genres import Genre, get_genre
from .midi_io import Note, Track
from .theory import (
    CHORD_TYPES, NOTE_NAMES, SCALES, chord_midi, degree_to_chord, pent_for, scale_notes,
)

BEATS_PER_BAR = 4
KICK, SNARE, CLAP, HAT, OPEN_HAT, RIDE = 36, 38, 39, 42, 46, 51
CONGA_HI, CONGA_LO, SHAKER, CHING = 63, 64, 70, 81

# (section name, bars, intensity 0..1)
SECTION_TEMPLATE = [
    ("intro", 4, 0.35),
    ("verse", 8, 0.6),
    ("chorus", 8, 1.0),
    ("verse", 8, 0.65),
    ("chorus", 8, 1.0),
    ("outro", 4, 0.3),
]


@dataclass
class Song:
    genre: str
    bpm: float
    root_pc: int
    scale: str
    bars: int
    seed: int
    tracks: list[Track] = field(default_factory=list)
    sections: list[tuple[str, int, int, float]] = field(default_factory=list)  # name, start_bar, bars, intensity
    offset_s: float = 0.0  # music starts this many seconds in (aligned to the vocal's bar line)

    @property
    def key_name(self) -> str:
        return f"{NOTE_NAMES[self.root_pc]} {self.scale}"

    @property
    def seconds(self) -> float:
        return self.offset_s + self.bars * BEATS_PER_BAR * 60.0 / self.bpm


def build_sections(total_bars: int) -> list[tuple[str, int, int, float]]:
    """Scale the section template so the song is ~total_bars long."""
    base = sum(b for _, b, _ in SECTION_TEMPLATE)
    plan = []
    bar = 0
    for i, (name, bars, inten) in enumerate(SECTION_TEMPLATE):
        n = max(2, round(bars * total_bars / base / 2) * 2)
        plan.append((name, bar, n, inten))
        bar += n
    # trim / pad the last section so we hit total_bars exactly
    name, start, n, inten = plan[-1]
    plan[-1] = (name, start, max(1, total_bars - start), inten)
    plan = [p for p in plan if p[1] < total_bars]
    return plan


def _swing_offset(step: int, swing: float) -> float:
    """Beat offset for a 16th step (odd 16ths pushed late by swing)."""
    return 0.25 * swing if step % 2 == 1 else 0.0


def _drums(rng: random.Random, g: Genre, sections, bars: int) -> Track:
    tr = Track("Drums", is_drums=True)
    if g.extra.get("no_drums"):
        return tr
    for name, start, n, inten in sections:
        if name == "intro" and inten < 0.5:
            use_hat_only = True
        else:
            use_hat_only = False
        for b in range(start, start + n):
            bar_beat = b * BEATS_PER_BAR
            last_bar_of_section = b == start + n - 1
            for step in range(16):
                t = bar_beat + step * 0.25 + _swing_offset(step, g.swing)
                k, s, h = g.kick[step], g.snare[step], g.hat[step]
                human = rng.randint(-4, 4)
                if h != ".":
                    vel = (62 if step % 4 else 78) + human
                    if rng.random() < 0.08 and step % 4 == 2:
                        tr.notes.append(Note(t, 0.2, OPEN_HAT, 70))
                    else:
                        tr.notes.append(Note(t, 0.1, HAT, int(vel * (0.7 + 0.3 * inten))))
                if use_hat_only:
                    continue
                if k != ".":
                    tr.notes.append(Note(t, 0.15, KICK, 108 + human))
                elif inten > 0.7 and rng.random() < 0.06 and step % 2 == 0:
                    tr.notes.append(Note(t, 0.15, KICK, 80))  # ghost kick
                if s != ".":
                    tr.notes.append(Note(t, 0.15, SNARE, 104 + human))
                    if inten >= 1.0 and g.name in ("pop", "edm", "house"):
                        tr.notes.append(Note(t, 0.15, CLAP, 70))
                elif rng.random() < 0.05 * inten and step % 2 == 1:
                    tr.notes.append(Note(t, 0.1, SNARE, 45))  # ghost snare
            if last_bar_of_section and n >= 4 and inten > 0.5:
                for i in range(4):  # small fill on last beat
                    tr.notes.append(Note(bar_beat + 3 + i * 0.25, 0.12, SNARE, 70 + i * 10))
    return tr


def _bass(rng: random.Random, g: Genre, chords, sections, bars: int, guide=None) -> Track:
    tr = Track("Bass", program=38, channel=1)
    for name, start, n, inten in sections:
        if name == "intro" and g.bass_style == "sub" and inten < 0.4:
            pass
        for b in range(start, start + n):
            root_pc, kind = chords[b % len(chords)]
            root = 28 + ((root_pc - 28) % 12)  # E1..D#2 region
            tt = b * BEATS_PER_BAR
            tones = chord_midi(root_pc, kind, root)
            if g.bass_style == "root":
                tr.notes.append(Note(tt, 3.8, root, 96))
            elif g.bass_style == "octave":
                for i in range(8):
                    p = root + (12 if i % 2 else 0)
                    tr.notes.append(Note(tt + i * 0.5, 0.45, p, 92 if i % 2 == 0 else 78))
            elif g.bass_style == "walking":
                seq = [tones[0], tones[1 % len(tones)], tones[2 % len(tones)], tones[1 % len(tones)]]
                for i, p in enumerate(seq):
                    tr.notes.append(Note(tt + i, 0.92, p, 88))
            elif g.bass_style == "sub":
                tr.notes.append(Note(tt, 1.9, root, 100))
                tr.notes.append(Note(tt + 2, 1.9 if rng.random() < 0.6 else 0.9, root, 92))
            elif g.bass_style == "syncopated":
                sung = guide is not None and sum(0 if _free_fn(guide)(tt + k * 0.5) else 1 for k in range(8)) >= 4
                pattern = [0, 2.0, 3.5] if sung else [0, 0.75, 1.5, 2.5, 3.0, 3.5]
                for off in pattern:
                    if rng.random() < 0.9:
                        p = root if rng.random() < 0.7 else tones[2 % len(tones)]
                        tr.notes.append(Note(tt + off, 0.4, p, 94))
            else:  # arp
                for i in range(8):
                    tr.notes.append(Note(tt + i * 0.5, 0.45, tones[i % len(tones)], 86))
    return tr


def _chords(rng: random.Random, g: Genre, chords, sections, bars: int) -> Track:
    tr = Track("Chords", program=89 if g.chord_timbre == "pad" else 19, channel=2)
    for name, start, n, inten in sections:
        for b in range(start, start + n):
            root_pc, kind = chords[b % len(chords)]
            tones = chord_midi(root_pc, kind, 55)
            tt = b * BEATS_PER_BAR
            vel = int(60 + 30 * inten)
            style = g.chord_style
            if style == "pad":
                for p in tones:
                    tr.notes.append(Note(tt, 3.95, p, vel - 10))
            elif style == "stabs":
                for off in (0, 1.5, 2.5):
                    for p in tones:
                        tr.notes.append(Note(tt + off, 0.6, p, vel))
            elif style == "arp":
                order = tones + [tones[0] + 12, tones[1] + 12]
                for i in range(8):
                    tr.notes.append(Note(tt + i * 0.5, 0.5, order[i % len(order)], vel - 4))
            elif style == "strum":
                for beat in (0, 1, 2, 3):
                    for j, p in enumerate(tones):
                        tr.notes.append(Note(tt + beat + j * 0.02, 0.9, p, vel - (0 if beat % 2 == 0 else 10)))
            elif style == "offbeat":
                for beat in (0.5, 1.5, 2.5, 3.5):
                    for p in tones:
                        tr.notes.append(Note(tt + beat, 0.4, p, vel))
    return tr


def _melody(rng: random.Random, g: Genre, root_pc: int, scale: str, chords, sections, bars: int) -> Track:
    tr = Track("Lead", program=80 if g.lead_timbre in ("saw", "square") else 73, channel=3)
    lo, hi = g.melody_range
    pool = scale_notes(root_pc, scale, lo, hi)
    if not pool:
        return tr
    idx = len(pool) // 2
    motif: list[tuple[float, float, int]] = []  # (offset, dur, step) reused across bars
    for name, start, n, inten in sections:
        if name in ("intro", "outro") and inten < 0.4:
            density = g.melody_density * 0.4
        else:
            density = g.melody_density * (0.7 + 0.5 * inten)
        for b in range(start, start + n):
            tt = b * BEATS_PER_BAR
            root_pc_c, kind = chords[b % len(chords)]
            chord_pcs = {(root_pc_c + i) % 12 for i in (0, 4 if "maj" in kind or kind == "dom7" else 3, 7)}
            if (b - start) % 4 == 0 or not motif:
                motif = []
                pos = 0.0
                while pos < 4.0:
                    dur = rng.choice([0.5, 0.5, 1.0, 0.25, 0.75, 1.5])
                    dur = min(dur, 4.0 - pos)
                    if rng.random() < density:
                        motif.append((pos, dur, rng.choice([-2, -1, -1, 0, 1, 1, 2])))
                    pos += dur
            for off, dur, step in motif:
                idx = max(0, min(len(pool) - 1, idx + step))
                if off in (0.0, 2.0):  # strong beats lean on chord tones
                    for _ in range(3):
                        if pool[idx] % 12 in chord_pcs:
                            break
                        idx = max(0, min(len(pool) - 1, idx + rng.choice([-1, 1])))
                tr.notes.append(Note(tt + off, dur * 0.95, pool[idx], int(70 + 30 * inten + rng.randint(-5, 5))))
        # resolve to the tonic at the end of each section
        end = (start + n) * BEATS_PER_BAR - 2
        tonic = min(pool, key=lambda p: (abs(p - pool[idx]) if (p - root_pc) % 12 == 0 else 99))
        tr.notes.append(Note(end, 2.0, tonic, 80))
    return tr



# ---------------------------------------------------------------------------
# Vocal-aware helpers + Isan-indie arrangement
# ---------------------------------------------------------------------------

def _chord_pcs(chord: tuple[int, str]) -> set[int]:
    root, kind = chord
    return {(root + i) % 12 for i in CHORD_TYPES[kind]}


def _free_fn(guide):
    """free(t_beats) -> True when the vocal is silent at that moment (always True with no vocal)."""
    if guide is None:
        return lambda t: True
    act = guide.activity

    def free(t: float) -> bool:
        i = int(t * 2)
        return not (0 <= i < len(act) and bool(act[i]))
    return free


def _vocal_chords(root_pc, scale, guide, prog, bars, seventh):
    """Choose one chord per bar by Viterbi search: fit the sung notes, stay close to the genre's
    progression, avoid pointless changes and favour natural moves (4th/5th)."""
    cands = []
    for d in range(1, 8):
        c = degree_to_chord(root_pc, scale, d, seventh)
        if c[1] != "dim" and c not in cands:
            cands.append(c)
    base = [degree_to_chord(root_pc, scale, d, seventh) for d in prog]
    K = len(cands)
    emit = np.zeros((bars, K))
    for b in range(bars):
        want = base[b % len(base)]
        chroma = guide.bar_chroma[b] if b < len(guide.bar_chroma) else None
        sung = chroma is not None and chroma.sum() > 0
        for k, c in enumerate(cands):
            r, kind = c
            pcs = [(r + i) % 12 for i in CHORD_TYPES[kind]]
            if sung:
                sc = chroma[pcs[0]] * 1.0 + chroma[pcs[1]] * 0.85 + chroma[pcs[2]] * 0.85
                sc -= 0.45 * (1.0 - sum(chroma[p] for p in pcs))
                sc += 0.10 * (c == want)
            else:
                sc = 0.35 * (c == want)
            emit[b, k] = sc
    trans = np.zeros((K, K))
    for i, (r1, _) in enumerate(cands):
        for j, (r2, _) in enumerate(cands):
            iv = (r2 - r1) % 12
            trans[i, j] = 0.12 if i == j else (0.05 if iv in (5, 7) else (0.02 if iv in (2, 10, 3, 8) else 0.0))
    score = emit[0].copy()
    back = np.zeros((bars, K), dtype=int)
    for b in range(1, bars):
        cand = score[:, None] + trans
        back[b] = np.argmax(cand, axis=0)
        score = cand[back[b], np.arange(K)] + emit[b]
    path = [int(np.argmax(score))]
    for b in range(bars - 1, 0, -1):
        path.append(int(back[b][path[-1]]))
    return [cands[k] for k in reversed(path)]


def _bar_info(sections, bars, guide=None):
    """(section name, intensity) per bar. With a vocal, intensity follows how hard the singer sings."""
    info = [("verse", 0.6)] * bars
    for name, start, n, inten in sections:
        for b in range(start, min(bars, start + n)):
            info[b] = (name, inten)
    if guide is None:
        return info
    e = np.asarray(guide.energy[:bars], dtype=float)
    k = np.array([0.25, 0.5, 0.25])
    sm = np.convolve(np.pad(e, 1, mode="edge"), k, mode="valid")
    out = []
    for b, (name, inten) in enumerate(info):
        if b < len(sm) and e[b] > 0.05:
            mixed = 0.5 * inten + 0.5 * (0.3 + 0.7 * sm[b])
            out.append((name, float(np.clip(mixed, 0.25, 1.0))))
        else:
            out.append((name, inten * 0.9))   # instrumental bar: follow the section plan
    return out


def _snap(pool, midi):
    """Nearest note of `pool` to `midi` (after folding by octaves into the pool range)."""
    lo, hi = pool[0], pool[-1]
    while midi < lo:
        midi += 12
    while midi > hi:
        midi -= 12
    return min(pool, key=lambda p: abs(p - midi))


def _echo(guide, slot, pool, free, max_notes=4, up=0):
    """Answer the phrase the singer just finished: same contour (snapped to the pentatonic pool), in the gap.
    Returns [(t_beats, dur_beats, midi)]."""
    if guide is None:
        return []
    ph = guide.recent_phrase(slot, 8)
    if len(ph) < 2:
        return []
    ph = ph[-max_notes:]
    base = ph[0][0]
    out = []
    for rel, ln, m in ph:
        t = slot * 0.5 + (rel - base) * 0.5
        if not free(t):
            break
        out.append((t, max(0.4, ln * 0.5 * 0.95), _snap(pool, m + up)))
    return out


def _isan_layers(rng, g, root_pc, scale, chords, sections, bars, guide) -> list[Track]:
    """Isan-indie band: phin, khaen, wot, pong lang, guitar, piano, synth pad, percussion."""
    free = _free_fn(guide)
    info = _bar_info(sections, bars, guide)
    pent = pent_for(scale)
    ch = lambda b: chords[b % len(chords)]  # noqa: E731

    def clampi(i, pool):
        return max(0, min(len(pool) - 1, i))

    def voiced_share(b):
        return sum(0 if free(b * 4 + k * 0.5) else 1 for k in range(8)) / 8.0

    reg = guide.register if guide is not None else None

    def under(ps, share):
        """Keep chord voicings below the singer's register while they are singing."""
        ps = list(ps)
        if reg is None or share < 0.35:
            return ps
        while max(ps) > reg - 2 and min(ps) - 12 >= 36:
            ps = [p - 12 for p in ps]
        return ps

    # ---- Phin: plucked ostinato in the gaps, a single anchor note under the singer
    phin = Track("Phin", program=105, channel=4)
    pool = scale_notes(root_pc, pent, 55, 79)
    idx = len(pool) // 2
    acc = [1, 0, .6, .5, .8, 0, .6, .4, 1, 0, .6, .5, .8, .5, .6, .4]
    phin_skip = -1.0
    for b in range(bars):
        name, inten = info[b]
        dens = 0.25 + 0.6 * inten
        pcs = _chord_pcs(ch(b))
        for s in range(16):
            t = b * 4 + s * 0.25
            if t < phin_skip:
                continue
            if guide is not None and s % 2 == 0 and b * 8 + s // 2 > 0 and free(t) and not free(t - 0.5) and rng.random() < 0.5:
                ans = _echo(guide, b * 8 + s // 2, pool, free, max_notes=4, up=0)
                if ans:
                    for tt_, d_, m_ in ans:
                        phin.notes.append(Note(tt_ - 0.125, 0.1, m_ - 2, 50))      # slide-in grace note
                        phin.notes.append(Note(tt_, min(d_, 0.6), m_, int(72 + 20 * inten)))
                    phin_skip = ans[-1][0] + ans[-1][1]
                    continue
            if not free(t):
                if s == 0 and rng.random() < 0.5:
                    r = 48 + ((ch(b)[0] - 48) % 12)
                    phin.notes.append(Note(t, 1.5, r + 12, 52))
                continue
            if rng.random() < acc[s] * dens:
                idx = clampi(idx + rng.choice([-2, -1, -1, 0, 1, 1, 2]), pool)
                if s % 4 == 0:
                    for _ in range(3):
                        if pool[idx] % 12 in pcs:
                            break
                        idx = clampi(idx + rng.choice([-1, 1]), pool)
                if rng.random() < 0.15 and idx > 0 and s > 0:
                    phin.notes.append(Note(t - 0.125, 0.1, pool[idx - 1], 55))
                phin.notes.append(Note(t, 0.22, pool[idx], int(68 + 25 * inten * acc[s] + rng.randint(-5, 5))))

    # ---- Khaen: reed drone (root + fifth) with rhythmic chops in the chorus
    khaen = Track("Khaen", program=21, channel=5)
    for b in range(bars):
        name, inten = info[b]
        r = 40 + ((ch(b)[0] - 40) % 12)
        soft = 0.7 if voiced_share(b) > 0.5 else 1.0
        vel = int((42 + 25 * inten) * soft)
        tt = b * 4
        for p in (r, r + 7, r + 12):
            khaen.notes.append(Note(tt, 3.9, p, vel))
        if inten >= 0.9:
            tones = [p for p in scale_notes(root_pc, pent, 60, 76) if p % 12 in _chord_pcs(ch(b))][:3]
            for off in (1.5, 3.5):
                if free(tt + off):
                    for p in tones:
                        khaen.notes.append(Note(tt + off, 0.4, p, 62))

    # ---- Wot: breathy panpipe phrases that answer the vocal (only in the silences)
    wot = Track("Wot", program=75, channel=6)
    pool = scale_notes(root_pc, pent, 67, 91)
    idx = len(pool) // 2
    left = rest = 0
    for b in range(bars):
        name, inten = info[b]
        prob = 0.3 if name == "intro" else 0.2 + 0.45 * inten
        for s in range(8):
            t = b * 4 + s * 0.5
            if not free(t):
                left = 0
                continue
            if rest > 0:
                rest -= 1
                continue
            if left == 0 and guide is not None and b * 8 + s > 0 and not free(t - 0.5) and rng.random() < 0.75:
                ans = _echo(guide, b * 8 + s, pool, free, max_notes=4, up=12)
                if ans:   # answer the singer's phrase with its own contour
                    for tt_, d_, m_ in ans:
                        wot.notes.append(Note(tt_, d_, m_, int(64 + 20 * inten)))
                    rest = max(2, int(round((ans[-1][0] + ans[-1][1] - t) * 2)) - 1 + rng.randint(1, 3))
                    continue
            if left == 0:
                if rng.random() < prob:
                    left = rng.randint(3, 5)
                else:
                    continue
            idx = clampi(idx + rng.choice([-3, -2, -1, 1, 2, 3]), pool)
            dur = 0.95 if (left == 1 and free(t + 0.5)) else 0.45
            wot.notes.append(Note(t, dur, pool[idx], int(62 + 20 * inten)))
            left -= 1
            if left == 0:
                rest = rng.randint(2, 5)

    # ---- Guitar: clean arpeggios (verse) / strums (chorus)
    gtr = Track("Guitar", program=27, channel=7)
    for b in range(bars):
        name, inten = info[b]
        r, kind = ch(b)
        tones = under(chord_midi(r, kind, 52), voiced_share(b))
        arp = tones + [tones[0] + 12]
        vel = int((48 + 28 * inten) * (0.75 if voiced_share(b) > 0.5 else 1.0))
        tt = b * 4
        if inten >= 0.9:
            for beat in (0, 1, 1.5, 2.5, 3):
                for j, p in enumerate(tones):
                    gtr.notes.append(Note(tt + beat + j * 0.015, 0.9, p, vel))
        else:
            for i, k in enumerate((0, 2, 1, 2, 0, 2, 1, 2)):
                gtr.notes.append(Note(tt + i * 0.5, 0.5, arp[k % len(arp)], vel - (6 if i % 2 else 0)))

    # ---- Piano: open chords on 1 and 3, small right-hand fills in free space
    pno = Track("Piano", program=0, channel=8)
    fill_pool = scale_notes(root_pc, pent, 72, 91)
    for b in range(bars):
        name, inten = info[b]
        r, kind = ch(b)
        tones = chord_midi(r, kind, 48)
        voicing = under([tones[0], tones[2], tones[1] + 12], voiced_share(b))
        vel = int((46 + 22 * inten) * (0.75 if voiced_share(b) > 0.5 else 1.0))
        tt = b * 4
        for beat in (0, 2):
            for p in voicing:
                pno.notes.append(Note(tt + beat, 1.9, p, vel))
        if fill_pool and name != "intro":
            for s in range(1, 8, 2):
                if free(tt + s * 0.5) and rng.random() < 0.18 * inten:
                    pno.notes.append(Note(tt + s * 0.5, 0.45, rng.choice(fill_pool), 56))

    # ---- Synth pad underneath everything
    syn = Track("Synth", program=89, channel=10)
    for b in range(bars):
        name, inten = info[b]
        r, kind = ch(b)
        for p in under(chord_midi(r, kind, 60), voiced_share(b)):
            syn.notes.append(Note(b * 4, 3.95, p, int(34 + 20 * inten)))

    # ---- Pong lang (wooden xylophone) ostinato in verses/choruses, in the gaps only
    pong = Track("Pong", program=13, channel=11)
    pool = scale_notes(root_pc, pent, 64, 88)
    motif = [0, 2, 1, 3]
    if pool:
        base = len(pool) // 3
        for b in range(bars):
            name, inten = info[b]
            if inten < 0.5:
                continue
            for s in range(8):
                t = b * 4 + s * 0.5
                if free(t) and (s % 2 == 0 or rng.random() < 0.4):
                    pong.notes.append(Note(t, 0.3, pool[clampi(base + motif[s % 4], pool)], int(58 + 15 * inten)))

    # ---- Percussion: ching off-beats, congas, shaker
    perc = Track("Perc", is_drums=True)
    for b in range(bars):
        name, inten = info[b]
        soft = 0.7 if voiced_share(b) > 0.5 else 1.0
        tt = b * 4
        for off in (0.5, 1.5, 2.5, 3.5):
            if rng.random() < 0.92:
                perc.notes.append(Note(tt + off, 0.1, CHING, int(62 * soft)))
        for s in range(0, 16, 2):
            perc.notes.append(Note(tt + s * 0.25, 0.05, SHAKER, int((44 + (8 if s % 4 == 0 else 0)) * soft)))
        if inten >= 0.5:
            for s in (2, 5, 10, 13):
                perc.notes.append(Note(tt + s * 0.25, 0.15, CONGA_HI, int(70 * soft)))
            for s in (7, 15):
                perc.notes.append(Note(tt + s * 0.25, 0.15, CONGA_LO, int(74 * soft)))

    return [phin, khaen, wot, gtr, pno, syn, pong, perc]


def compose(
    genre: str = "all",
    bpm: float | None = None,
    key: int | None = None,
    scale: str | None = None,
    bars: int = 32,
    seed: int | None = None,
    lead: bool = True,
    guide=None,
) -> Song:
    """Compose a song. All arguments optional; `genre='all'` picks a random genre."""
    if seed is None:
        seed = random.randrange(1, 2**31)
    rng = random.Random(seed)
    g = get_genre(genre, rng)
    bpm = float(bpm) if bpm else float(rng.randint(*g.bpm))
    scale = scale if scale in SCALES else rng.choice(g.scales)
    root_pc = key if key is not None else rng.randrange(12)
    prog = rng.choice(g.progressions)
    if guide is not None:
        chords = _vocal_chords(root_pc, scale, guide, prog, bars, g.seventh)
    else:
        chords = [degree_to_chord(root_pc, scale, d, g.seventh) for d in prog]

    sections = build_sections(bars)
    song = Song(g.name, bpm, root_pc, scale, bars, seed, sections=sections)
    song.tracks.append(_drums(rng, g, sections, bars))
    song.tracks.append(_bass(rng, g, chords, sections, bars, guide))
    if g.extra.get("isan"):
        song.tracks.extend(_isan_layers(rng, g, root_pc, scale, chords, sections, bars, guide))
    else:
        song.tracks.append(_chords(rng, g, chords, sections, bars))
        if lead:
            song.tracks.append(_melody(rng, g, root_pc, scale, chords, sections, bars))
    song.tracks = [t for t in song.tracks if t.notes]
    return song
