from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
import pickle
import os

app = Flask(__name__, static_folder="static", template_folder="templates")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
ENC_PATH = os.path.join(BASE_DIR, "encoders.pkl")

model = None
le_bat = le_bowl = le_venue = None


def load_model():
    """Load model and encoders once (lazy loading)."""
    global model, le_bat, le_bowl, le_venue

    if model is None:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(ENC_PATH):
            raise RuntimeError(
                "Model files not found. Ensure train_and_save.py ran successfully."
            )

        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        with open(ENC_PATH, "rb") as f:
            encoders = pickle.load(f)

        le_bat = encoders["bat"]
        le_bowl = encoders["bowl"]
        le_venue = encoders["venue"]

        print("✅ Model and encoders loaded.")


@app.route("/")
def index():
    load_model()
    teams = list(le_bat.classes_)
    venues = list(le_venue.classes_)
    return render_template("index.html", teams=teams, venues=venues)


@app.route("/predict", methods=["POST"])
def predict():
    load_model()

    data = request.json
    required = [
        "bat_team",
        "bowl_team",
        "venue",
        "overs",
        "runs_last_5",
        "wickets_last_5",
    ]

    for r in required:
        if r not in data:
            return jsonify({"error": f"Missing field '{r}'"}), 400

    try:
        bat = int(le_bat.transform([data["bat_team"]])[0])
        bowl = int(le_bowl.transform([data["bowl_team"]])[0])
        venue = int(le_venue.transform([data["venue"]])[0])

        inp = np.array([[
            bat,
            bowl,
            venue,
            float(data["overs"]),
            int(data["runs_last_5"]),
            int(data["wickets_last_5"]),
        ]])

    except Exception:
        return jsonify({"error": "Invalid input values."}), 400

    tree_preds = np.array([est.predict(inp)[0] for est in model.estimators_])

    mean_pred = float(np.mean(tree_preds))
    median_pred = float(np.median(tree_preds))

    bin_width = int(data.get("bin_width", 10))
    min_score = max(0, int(tree_preds.min()) - bin_width)
    max_score = int(tree_preds.max()) + bin_width

    bins = list(range(min_score, max_score + bin_width, bin_width))
    counts, edges = np.histogram(tree_preds, bins=bins)

    labels, probs = [], []
    total = counts.sum() if counts.sum() else 1

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
        "range_low": max(0, round(mean_pred - np.std(tree_preds), 1)),
        "range_high": round(mean_pred + np.std(tree_preds), 1),
    })


@app.route("/static/<path:filename>")
def staticfiles(filename):
    return send_from_directory("static", filename)


# ❌ DO NOT use debug=True in production
if __name__ == "__main__":
    app.run()
