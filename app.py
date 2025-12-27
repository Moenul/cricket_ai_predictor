from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
import pickle
import os

app = Flask(__name__, static_folder="static", template_folder="templates")

# Use absolute paths to ensure the app finds files on the server
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
ENC_PATH = os.path.join(BASE_DIR, "encoders.pkl")

# Initialize global variables
model = None
le_bat = le_bowl = le_venue = None

def load_model():
    """Load model and encoders once (lazy loading) with error logging."""
    global model, le_bat, le_bowl, le_venue

    if model is None:
        # Check if the files actually exist before trying to open them
        if not os.path.exists(MODEL_PATH) or not os.path.exists(ENC_PATH):
            # This helps you see what's actually in the folder via logs
            files_in_dir = os.listdir(BASE_DIR)
            raise RuntimeError(
                f"Model files not found. Files present: {files_in_dir}. "
                "Ensure train_and_save.py ran successfully."
            )

        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        with open(ENC_PATH, "rb") as f:
            encoders = pickle.load(f)

        le_bat = encoders["bat"]
        le_bowl = encoders["bowl"]
        le_venue = encoders["venue"]

        print("✅ Model and encoders loaded successfully.")

@app.route("/")
def index():
    try:
        load_model()
        # Sort classes alphabetically for a better UI experience
        teams = sorted(list(le_bat.classes_))
        venues = sorted(list(le_venue.classes_))
        return render_template("index.html", teams=teams, venues=venues)
    except Exception as e:
        return f"Error loading application: {str(e)}", 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        load_model()
        data = request.json
        
        # 1. Validation
        required = ["bat_team", "bowl_team", "venue", "overs", "runs_last_5", "wickets_last_5"]
        for r in required:
            if r not in data:
                return jsonify({"error": f"Missing field '{r}'"}), 400

        # 2. Encoding & Preprocessing
        bat = int(le_bat.transform([data["bat_team"]])[0])
        bowl = int(le_bowl.transform([data["bowl_team"]])[0])
        venue = int(le_venue.transform([data["venue"]])[0])

        inp = np.array([[
            bat, bowl, venue,
            float(data["overs"]),
            int(data["runs_last_5"]),
            int(data["wickets_last_5"]),
        ]])

        # 3. Prediction Logic (Optimized)
        # Assuming RandomForestRegressor: Extracting predictions from individual trees
        tree_preds = np.array([est.predict(inp)[0] for est in model.estimators_])

        mean_pred = float(np.mean(tree_preds))
        median_pred = float(np.median(tree_preds))
        std_dev = np.std(tree_preds)

        # 4. Histogram/Probability Distribution
        bin_width = int(data.get("bin_width", 10))
        min_score = max(0, int(tree_preds.min()) - bin_width)
        max_score = int(tree_preds.max()) + bin_width

        bins = list(range(min_score, max_score + bin_width, bin_width))
        counts, edges = np.histogram(tree_preds, bins=bins)

        labels, probs = [], []
        total = counts.sum() if counts.sum() > 0 else 1

        for i in range(len(counts)):
            low = int(edges[i])
            high = int(edges[i + 1]) - 1
            labels.append(f"{low}-{high}")
            probs.append(round(float(counts[i]) / total, 4))

        return jsonify({
            "mean": round(mean_pred, 1),
            "median": round(median_pred, 1),
            "bins": labels,
            "probs": probs,
            "range_low": max(0, round(mean_pred - std_dev, 1)),
            "range_high": round(mean_pred + std_dev, 1),
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 400

@app.route("/static/<path:filename>")
def staticfiles(filename):
    return send_from_directory("static", filename)

if __name__ == "__main__":
    # Get port from environment variable for Railway/Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)