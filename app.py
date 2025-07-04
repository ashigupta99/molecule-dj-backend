from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os, tempfile, pickle
import tensorflow as tf
import numpy as np
from midiutil import MIDIFile
from utils import extract_features, describe_molecule, generate_music, explain_music_mapping

# Initialize app and set CORS to your exact frontend domain
app = Flask(__name__)
CORS(app, origins=[
    "https://molecule-dj-frontend.vercel.app"  # <-- CHANGE THIS to your real frontend URL
])

# Load model and scaler once at startup
model = tf.keras.models.load_model("molecule_dj_model_full.keras")
with open("scaler_full.pkl", "rb") as f:
    scaler = pickle.load(f)

# Music Generation Endpoint
@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    smiles = data.get("smiles", "")
    features = extract_features(smiles)

    if features is None:
        return jsonify({"error": "Invalid SMILES"}), 400

    notes = generate_music(model, smiles, features)
    explanation = explain_music_mapping(smiles, features, notes)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mid", dir=".")
    midi = MIDIFile(1)
    midi.addTempo(0, 0, 120)
    for i, note in enumerate(notes):
        midi.addNote(0, 0, note, i * 0.5, 1, 100)
    with open(temp_file.name, "wb") as f:
        midi.writeFile(f)

    return jsonify({
        "notes": notes,
        "explanation": explanation,
        "midi_url": f"/get-midi/{os.path.basename(temp_file.name)}"
    })

# Serve generated MIDI
@app.route("/get-midi/<filename>")
def get_midi(filename):
    path = os.path.join(".", filename)
    return send_file(path, as_attachment=True, download_name="generated.mid")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
