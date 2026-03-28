from flask import Flask, request
from flask_cors import CORS
from pathlib import Path

app = Flask(__name__)
CORS(app)
BASE_DIR = Path(__file__).resolve().parent
LOCATION_FILE = BASE_DIR / "location.txt"

@app.route('/save_location', methods=['POST'])
def save_location():
    data = request.json

    with open(LOCATION_FILE, "w", encoding="utf-8") as f:
        f.write(f"{data['lat']},{data['lng']}")

    print("Saved:", data, "->", LOCATION_FILE)

    return "Saved"

app.run(port=5000)
