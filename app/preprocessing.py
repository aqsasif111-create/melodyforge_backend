"""
Step 1 + 2: Collect MIDI data & preprocess into note sequences.

We use music21 to parse MIDI files into a flat stream of notes/chords,
encode each as a string token (e.g. "C4" for a note, "4.7.0" for a chord
by pitch class), and cache the tokenized corpus + vocabulary to disk so
training doesn't have to re-parse MIDI every run.
"""
import glob
import os
import pickle

import numpy as np
from music21 import converter, instrument, note, chord

from . import config


def _file_to_tokens(midi_path: str) -> list[str]:
    """Parse a single MIDI file into a list of note/chord tokens."""
    tokens: list[str] = []
    try:
        score = converter.parse(midi_path)
    except Exception as e:
        print(f"[preprocessing] skipping {midi_path}: {e}")
        return tokens

    # Flatten to a single part (piano reduction) so polyphonic scores
    # still produce one sequential token stream.
    parts = instrument.partitionByInstrument(score)
    stream = parts.parts[0].recurse() if parts else score.flat.notes

    for element in stream:
        if isinstance(element, note.Note):
            tokens.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            tokens.append(".".join(str(n) for n in element.normalOrder))
    return tokens


def build_corpus(genre: str) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """
    Collect Step: walk DATASET_DIR/<genre>/*.mid, tokenize every file.
    Preprocess Step: build the vocabulary and int-encoded corpus.
    Caches results to PROCESSED_DIR/<genre>.pkl so re-runs are instant.
    """
    cache_path = os.path.join(config.PROCESSED_DIR, f"{genre}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    midi_files = glob.glob(os.path.join(config.DATASET_DIR, genre, "*.mid"))
    if not midi_files:
        raise FileNotFoundError(
            f"No MIDI files found for genre '{genre}' in {config.DATASET_DIR}/{genre}/"
        )

    all_tokens: list[str] = []
    for path in midi_files:
        all_tokens.extend(_file_to_tokens(path))

    vocab = sorted(set(all_tokens))
    token_to_int = {tok: i for i, tok in enumerate(vocab)}
    int_to_token = {i: tok for tok, i in token_to_int.items()}

    with open(cache_path, "wb") as f:
        pickle.dump((all_tokens, token_to_int, int_to_token), f)

    return all_tokens, token_to_int, int_to_token


def make_training_sequences(
    tokens: list[str], token_to_int: dict[str, int], seq_len: int = config.SEQUENCE_LENGTH
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window of seq_len tokens across the corpus to build (X, y) pairs."""
    encoded = [token_to_int[t] for t in tokens]
    X, y = [], []
    for i in range(len(encoded) - seq_len):
        X.append(encoded[i : i + seq_len])
        y.append(encoded[i + seq_len])

    n_vocab = len(token_to_int)
    X = np.reshape(X, (len(X), seq_len, 1)) / float(n_vocab)
    y = np.eye(n_vocab)[y]  # one-hot
    return X, y
