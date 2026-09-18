"""
Step 5 (part 2): Convert generated note sequences back into a MIDI file
so the frontend can play/download it.
"""
import os
import uuid

from music21 import stream, note, chord, tempo

from . import config


def tokens_to_midi(tokens: list[str], bpm: int = 120) -> str:
    """Turn a list of note/chord tokens into a .mid file, return its path."""
    output_stream = stream.Stream()
    output_stream.append(tempo.MetronomeMark(number=bpm))

    for token in tokens:
        if "." in token:
            # chord: token looks like "4.7.0" (pitch classes)
            notes_in_chord = [note.Note(int(n)) for n in token.split(".")]
            output_stream.append(chord.Chord(notes_in_chord))
        else:
            output_stream.append(note.Note(token))

    filename = f"{uuid.uuid4().hex}.mid"
    out_path = os.path.join(config.GENERATED_DIR, filename)
    output_stream.write("midi", fp=out_path)
    return out_path
