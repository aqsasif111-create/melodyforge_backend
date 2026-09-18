"""
MelodyForge AI backend API.

Maps directly onto the app's screens:
  Home / Generate screen  -> POST /generate
  Training (offline/admin)-> POST /train
  Library screen           -> GET /tracks, GET /tracks/{id}/download
  Genre chips               -> GET /genres

Run locally with:
    uvicorn app.main:app --reload --port 8000
"""
import os
import time
import uuid
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import config
from .midi_utils import tokens_to_midi
from .model import generate_sequence, train

app = FastAPI(title="MelodyForge AI API", version="1.0.0")

# Allow the Framer/mobile app's domain to call this API directly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your app's actual domain(s) in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory track registry for demo purposes; swap for a real DB (Postgres, etc.) in production.
TRACKS: dict[str, dict] = {}


# ---------- Schemas ----------

class GenerateRequest(BaseModel):
    genre: Literal["classical", "jazz", "pop", "lofi", "ambient"]
    mood: float = Field(0.5, ge=0.0, le=1.0, description="0 = calm, 1 = energetic")
    tempo_bpm: int = Field(120, ge=40, le=220)
    duration_notes: int = Field(200, ge=20, le=1000)


class TrainRequest(BaseModel):
    genre: Literal["classical", "jazz", "pop", "lofi", "ambient"]


class TrackOut(BaseModel):
    id: str
    title: str
    genre: str
    created_at: float
    download_url: str


# ---------- Endpoints ----------

@app.get("/genres")
def list_genres():
    """Populates the genre chips / dropdown on the Home & Generate screens."""
    return {"genres": config.GENRES}


@app.post("/train")
def train_model(req: TrainRequest, background_tasks: BackgroundTasks):
    """
    Kicks off Steps 1-4 (collect -> preprocess -> build -> train) for a genre.
    Training is slow, so it runs in the background; poll /tracks or a
    separate /train/status/{genre} endpoint (add a job table for that
    in production) to check progress.
    """
    background_tasks.add_task(train, req.genre)
    return {"status": "training_started", "genre": req.genre}


@app.post("/generate", response_model=TrackOut)
def generate_track(req: GenerateRequest):
    """
    Step 5: generate a new note sequence with the trained model, convert
    it to MIDI, and register it so it shows up in the Library screen.
    Powers the "Generate" button on the mobile app.
    """
    try:
        tokens = generate_sequence(
            genre=req.genre, length=req.duration_notes, mood=req.mood
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    midi_path = tokens_to_midi(tokens, bpm=req.tempo_bpm)

    track_id = uuid.uuid4().hex
    track = {
        "id": track_id,
        "title": f"{req.genre.title()} Piece #{len(TRACKS) + 1}",
        "genre": req.genre,
        "created_at": time.time(),
        "file_path": midi_path,
        "download_url": f"/tracks/{track_id}/download",
    }
    TRACKS[track_id] = track
    return track


@app.get("/tracks", response_model=list[TrackOut])
def list_tracks(genre: str | None = None):
    """Powers the Library screen, with optional genre filter."""
    tracks = list(TRACKS.values())
    if genre:
        tracks = [t for t in tracks if t["genre"] == genre]
    tracks.sort(key=lambda t: t["created_at"], reverse=True)
    return tracks


@app.get("/tracks/{track_id}/download")
def download_track(track_id: str):
    """Serves the actual .mid file for playback/download in the app."""
    track = TRACKS.get(track_id)
    if not track or not os.path.exists(track["file_path"]):
        raise HTTPException(status_code=404, detail="Track not found")
    return FileResponse(
        track["file_path"],
        media_type="audio/midi",
        filename=f"{track['title']}.mid",
    )
