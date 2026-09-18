"""
Step 3: Build the deep learning model (stacked LSTM).
Step 4: Train the model on the tokenized dataset.
Step 5 (part 1): Generate new note-token sequences from a trained model.

A GAN variant is noted at the bottom for teams that want adversarial
training instead of the simpler next-token LSTM approach; the LSTM is
the practical default since it's far easier to train stably on modest
MIDI datasets.
"""
import os

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from . import config
from .preprocessing import build_corpus, make_training_sequences


def build_lstm_model(seq_len: int, n_vocab: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(seq_len, 1)),
        layers.LSTM(config.LSTM_UNITS, return_sequences=True),
        layers.Dropout(config.DROPOUT),
        layers.LSTM(config.LSTM_UNITS),
        layers.Dense(config.LSTM_UNITS, activation="relu"),
        layers.Dropout(config.DROPOUT),
        layers.Dense(n_vocab, activation="softmax"),
    ])
    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
    )
    return model


def train(genre: str) -> str:
    """Train (or continue training) a per-genre model and save it to disk."""
    tokens, token_to_int, int_to_token = build_corpus(genre)
    X, y = make_training_sequences(tokens, token_to_int)

    model_path = os.path.join(config.MODEL_DIR, f"{genre}.keras")
    if os.path.exists(model_path):
        model = keras.models.load_model(model_path)
    else:
        model = build_lstm_model(config.SEQUENCE_LENGTH, len(token_to_int))

    checkpoint = keras.callbacks.ModelCheckpoint(
        model_path, monitor="loss", save_best_only=True
    )
    early_stop = keras.callbacks.EarlyStopping(monitor="loss", patience=5)

    model.fit(
        X, y,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=[checkpoint, early_stop],
    )
    model.save(model_path)
    return model_path


def _mood_to_temperature(mood: float) -> float:
    """Map a 0-1 'calm -> energetic' UI slider to a softmax sampling temperature."""
    lo, hi = config.TEMPERATURE_RANGE
    return lo + mood * (hi - lo)


def generate_sequence(genre: str, length: int = config.DEFAULT_GEN_LENGTH, mood: float = 0.5) -> list[str]:
    """
    Step 5 (part 1): sample a new token sequence from the trained model.
    `mood` in [0,1] controls sampling temperature -> randomness/energy.
    """
    tokens, token_to_int, int_to_token = build_corpus(genre)
    model_path = os.path.join(config.MODEL_DIR, f"{genre}.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model for genre '{genre}'. Run train() first.")

    model = keras.models.load_model(model_path)
    n_vocab = len(token_to_int)
    temperature = _mood_to_temperature(mood)

    # seed the generation with a random window from the real corpus
    encoded = [token_to_int[t] for t in tokens]
    start = np.random.randint(0, len(encoded) - config.SEQUENCE_LENGTH - 1)
    pattern = encoded[start : start + config.SEQUENCE_LENGTH]

    generated: list[str] = []
    for _ in range(length):
        x = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        probs = model.predict(x, verbose=0)[0]

        # temperature-scaled sampling for controllable "mood"
        probs = np.log(probs + 1e-9) / temperature
        probs = np.exp(probs) / np.sum(np.exp(probs))
        next_idx = np.random.choice(len(probs), p=probs)

        generated.append(int_to_token[next_idx])
        pattern.append(next_idx)
        pattern = pattern[1:]

    return generated


# --- Optional GAN alternative (Step 3, adversarial variant) ---
# A simple sequence GAN (e.g. C-RNN-GAN or a Transformer-based discriminator)
# can replace the LSTM above: a generator LSTM produces token sequences,
# a discriminator LSTM classifies real vs. generated sequences, and the
# two train adversarially. This tends to need much more data and tuning
# to converge, so it's recommended only once the LSTM baseline works well.
