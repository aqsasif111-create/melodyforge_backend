# MelodyForge AI — Backend

Backend for the MelodyForge AI music generation app. Implements the full pipeline:

| Task requirement | Where it lives |
|---|---|
| 1. Collect MIDI training data | `data/midi/<genre>/*.mid` (you supply the files, e.g. from [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/) or [MAESTRO](https://magenta.tensorflow.org/datasets/maestro)) |
| 2. Preprocess into note sequences | `app/preprocessing.py` (uses `music21`) |
| 3. Build LSTM/GAN model | `app/model.py::build_lstm_model` |
| 4. Train the model | `app/model.py::train` |
| 5. Generate & convert back to MIDI | `app/model.py::generate_sequence` + `app/midi_utils.py::tokens_to_midi` |

The FastAPI app in `app/main.py` wraps all of this in an HTTP API the Framer
website / mobile app can call directly.

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Add training data

Drop `.mid` files into genre subfolders:

```
data/midi/
  classical/
    song1.mid
    song2.mid
  jazz/
    ...
```

## Train a genre model

```bash
python -c "from app.model import train; train('classical')"
```

Or via the API once it's running (see below):

```bash
curl -X POST http://localhost:8000/train -H "Content-Type: application/json" \
  -d '{"genre": "classical"}'
```

Training runs in the background; check server logs for progress. Expect this
to take anywhere from minutes to hours depending on dataset size and hardware
— a GPU is strongly recommended.

## Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

Docs auto-generated at `http://localhost:8000/docs`.

## Key endpoints (map to app screens)

- `GET /genres` — genre chips on Home/Generate screens
- `POST /generate` — the big "Generate" button; body: `{genre, mood, tempo_bpm, duration_notes}`
- `GET /tracks` — Library screen (list of generated tracks)
- `GET /tracks/{id}/download` — serves the .mid file for playback/download

## Wiring into the frontend

From the Framer site or the mobile app, call `POST /generate`, then use the
returned `download_url` to fetch the MIDI file and either:
- play it client-side with a library like **Tone.js** (convert MIDI → audio in-browser), or
- render a simple piano-roll visualization of the note sequence.

## Notes / production hardening

- Swap the in-memory `TRACKS` dict for a real database (Postgres + an ORM like SQLAlchemy).
- Move generated `.mid` files to object storage (S3/GCS) rather than local disk.
- Add a `/train/status/{genre}` endpoint backed by a job table (or Celery/RQ) so the app can poll training progress instead of firing-and-forgetting.
- Restrict CORS `allow_origins` to your actual app domain(s) before shipping.
- Consider per-genre vocabulary caps (`NOTE_VOCAB_MIN_COUNT` in `config.py`) if a genre's dataset produces an unwieldy chord vocabulary.
