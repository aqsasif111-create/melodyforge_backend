"""
Central configuration for the MelodyForge AI backend.
Adjust paths / hyperparameters here rather than scattering magic numbers
throughout the codebase.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Data ---
DATASET_DIR = os.path.join(BASE_DIR, "data", "midi")          # raw MIDI files, one subfolder per genre
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")    # cached note-sequence .npy files
GENERATED_DIR = os.path.join(BASE_DIR, "data", "generated")    # output MIDI files served to the app
MODEL_DIR = os.path.join(BASE_DIR, "models")                   # saved .keras models, one per genre

for d in (DATASET_DIR, PROCESSED_DIR, GENERATED_DIR, MODEL_DIR):
    os.makedirs(d, exist_ok=True)

GENRES = ["classical", "jazz", "pop", "lofi", "ambient"]

# --- Preprocessing ---
SEQUENCE_LENGTH = 100        # how many previous notes the model sees before predicting the next
NOTE_VOCAB_MIN_COUNT = 1     # drop extremely rare tokens if you want a smaller vocabulary

# --- Model / training ---
LSTM_UNITS = 256
DROPOUT = 0.3
EPOCHS = 60
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# --- Generation ---
DEFAULT_GEN_LENGTH = 200      # number of notes to generate
TEMPERATURE_RANGE = (0.5, 1.3)  # maps "mood" slider (calm -> energetic) to sampling temperature
