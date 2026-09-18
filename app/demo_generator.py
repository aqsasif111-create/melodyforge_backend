"""
Demo / rule-based generator.

Produces musically plausible note sequences WITHOUT requiring a trained
LSTM model or a MIDI dataset. This lets the app work end-to-end immediately
(collect + preprocess + train are still implemented in preprocessing.py /
model.py for when real MIDI data is added later) while giving a real,
listenable result today.

Each genre gets a distinct scale, octave range, and chord density so the
five genres actually sound different from one another.
"""
import random

GENRE_PROFILES = {
    "classical": {"scale": [0, 2, 4, 5, 7, 9, 11], "octaves": (4, 5), "chord_chance": 0.15},  # major scale
    "jazz":      {"scale": [0, 2, 3, 5, 7, 9, 10], "octaves": (3, 5), "chord_chance": 0.35},  # dorian
    "pop":       {"scale": [0, 2, 4, 7, 9],        "octaves": (4, 5), "chord_chance": 0.20},  # major pentatonic
    "lofi":      {"scale": [0, 3, 5, 7, 10],       "octaves": (3, 4), "chord_chance": 0.10},  # minor pentatonic
    "ambient":   {"scale": [0, 2, 4, 6, 8, 10],    "octaves": (3, 5), "chord_chance": 0.05},  # whole tone
}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _pitch_class_to_name(pitch_class: int, octave: int) -> str:
    return f"{NOTE_NAMES[pitch_class % 12]}{octave}"


def generate_demo_sequence(genre: str, length: int = 200, mood: float = 0.5) -> list[str]:
    """
    Returns a token list in the SAME format the trained-model pipeline
    produces (plain note names like 'C4', or dot-separated pitch-class
    chords like '0.4.7'), so midi_utils.tokens_to_midi handles either
    source identically.

    `mood` in [0, 1]: calm -> energetic. Raises chord density and widens
    the octave range as mood increases.
    """
    profile = GENRE_PROFILES.get(genre, GENRE_PROFILES["pop"])
    scale = profile["scale"]
    lo, hi = profile["octaves"]
    chord_chance = min(0.9, profile["chord_chance"] + mood * 0.25)

    tokens: list[str] = []
    for _ in range(length):
        octave = random.randint(lo, hi + (1 if mood > 0.7 else 0))
        if random.random() < chord_chance:
            root = random.choice(scale)
            idx = scale.index(root)
            third = scale[(idx + 2) % len(scale)]
            fifth = scale[(idx + 4) % len(scale)]
            tokens.append(".".join(str(pc) for pc in sorted({root, third, fifth})))
        else:
            pitch_class = random.choice(scale)
            tokens.append(_pitch_class_to_name(pitch_class, octave))
    return tokens
