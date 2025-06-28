from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import os, tempfile, pickle
import numpy as np
import tensorflow as tf
from midiutil import MIDIFile
from utils import extract_features, describe_molecule, generate_music, explain_music_mapping

app = Flask(__name__, static_folder="frontend")

# CORS to allow your Vercel frontend
CORS(app, origins=["https://molecule-dj-frontend-r6ny8h0jj-aashi-guptas-projects-c296b8f3.vercel.app"])

# Lazy-loaded model and scaler
model = None
scaler = None

def load_model_and_scaler():
    global model, scaler
    if model is None:
        model = tf.keras.models.load_model("molecule_dj_model_full.keras")
    if scaler is None:
        with open("scaler_full.pkl", "rb") as f:
            scaler = pickle.load(f)

# MIDI conversion helper
def notes_to_midi(notes, path):
    midi = MIDIFile(1)
    midi.addTempo(0, 0, 120)
    for i, note in enumerate(notes):
        midi.addNote(0, 0, note, i * 0.5, 1, 100)
    with open(path, "wb") as f:
        midi.writeFile(f)

# Root for static frontend (optional)
@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")

# Music generation endpoint
@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    smiles = data.get("smiles", "").strip()
    
    features = extract_features(smiles)
    if features is None:
        return jsonify({"error": "Invalid SMILES"}), 400

    load_model_and_scaler()
    notes = generate_music(model, smiles, features)
    explanation = explain_music_mapping(smiles, features, notes)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mid", dir=".")
    notes_to_midi(notes, temp_file.name)
    temp_file.close()

    return jsonify({
        "notes": notes,
        "explanation": explanation,
        "midi_url": f"/get-midi/{os.path.basename(temp_file.name)}"
    })

# Health check for Render
@app.route("/ping")
def ping():
    return "pong", 200

# MIDI download
@app.route("/get-midi/<filename>")
def get_midi(filename):
    path = os.path.join(".", filename)
    return send_file(path, as_attachment=True, download_name="generated.mid")

# Serve static files from frontend (optional)
@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory("frontend", filename)

# Run on the port Render gives us
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
