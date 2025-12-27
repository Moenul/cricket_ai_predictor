from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
import pickle
import os

app = Flask(__name__, static_folder="static", template_folder="templates")

MODEL_PATH = "model.pkl"
ENC_PATH = "encoders.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(ENC_PATH):
    raise SystemExit("Run 'python train_and_save.py' first to create model.pkl and encoders.pkl")

model = pickle.load(open(MODEL_PATH, "rb"))
encoders = pickle.load(open(ENC_PATH, "rb"))
le_bat = encoders["bat"]
le_bowl = encoders["bowl"]
le_venue = encoders["venue"]

@app.route("/")
def index():
    teams = list(le_bat.classes_)
    venues = list(le_venue.classes_)
    return render_template("index.html", teams=teams, venues=venues)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    required = ["bat_team","bowl_team","venue","overs","runs_last_5","wickets_last_5"]
    for r in required:
        if r not in data:
            return jsonify({"error": f"Missing field '{r}'"}), 400
    try:
        bat = int(le_bat.transform([data["bat_team"]])[0])
        bowl = int(le_bowl.transform([data["bowl_team"]])[0])
        venue = int(le_venue.transform([data["venue"]])[0])
    except Exception as e:
        return jsonify({"error": "Team or venue not found in training encoders."}), 400

    inp = np.array([[bat, bowl, venue, float(data["overs"]), int(data["runs_last_5"]), int(data["wickets_last_5"])]])
    tree_preds = np.array([est.predict(inp)[0] for est in model.estimators_])
    mean_pred = float(np.mean(tree_preds))
    median_pred = float(np.median(tree_preds))

    bin_width = int(data.get("bin_width", 10))
    min_score = max(0, int(tree_preds.min()) - bin_width*2)
    max_score = int(tree_preds.max()) + bin_width*2
    bins = list(range(min_score, max_score + bin_width, bin_width))
    counts, edges = np.histogram(tree_preds, bins=bins)
    labels = []
    probs = []
    for i in range(len(counts)):
        low = int(edges[i])
        high = int(edges[i+1]) - 1
        labels.append(f"{low}-{high}")
        probs.append(float(counts[i]) / float(counts.sum()) if counts.sum() else 0.0)

    response = {
        "mean": mean_pred,
        "median": median_pred,
        "bins": labels,
        "probs": probs,
        "range_low": round(mean_pred - np.std(tree_preds), 1),
        "range_high": round(mean_pred + np.std(tree_preds), 1)
    }
    return jsonify(response)

@app.route('/static/<path:filename>')
def staticfiles(filename):
    return send_from_directory('static', filename)

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
import pickle
import os

app = Flask(__name__, static_folder="static", template_folder="templates")

MODEL_PATH = "model.pkl"
ENC_PATH = "encoders.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(ENC_PATH):
    raise SystemExit(
        "🚨 ERROR: Model or Encoders not found. Run 'python train_and_save.py' first to create model.pkl and encoders.pkl"
    )

try:
    model = pickle.load(open(MODEL_PATH, "rb"))
    encoders = pickle.load(open(ENC_PATH, "rb"))
    le_bat = encoders["bat"]
    le_bowl = encoders["bowl"]
    le_venue = encoders["venue"]
    print("✅ Model and Encoders loaded successfully.")
except Exception as e:
    raise SystemExit(f"❌ ERROR: Failed to load model or encoders. Details: {e}")

def get_frontend_data():
    """Extracts unique teams and venues from the loaded encoders."""
    teams = list(le_bat.classes_)
    venues = list(le_venue.classes_)
    return teams, venues


@app.route("/")
def index():
    """Home page route, supplies lists for dropdowns."""
    teams, venues = get_frontend_data()
    return render_template("index.html", teams=teams, venues=venues)

@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint to get prediction based on input data."""
    data = request.json
    required = ["bat_team","bowl_team","venue","overs","runs_last_5","wickets_last_5"]
    
    for r in required:
        if r not in data:
            return jsonify({"error": f"Missing field '{r}'"}), 400

    try:
        bat = int(le_bat.transform([data["bat_team"]])[0])
        bowl = int(le_bowl.transform([data["bowl_team"]])[0])
        venue = int(le_venue.transform([data["venue"]])[0])
    except Exception as e:
        return jsonify({"error": "Team or venue not found in training encoders. Please check your input."}), 400

    try:
        inp = np.array([[
            bat,
            bowl,
            venue,
            float(data["overs"]),
            int(data["runs_last_5"]),
            int(data["wickets_last_5"])
        ]])
    except ValueError:
        return jsonify({"error": "Overs, runs, or wickets must be valid numbers."}), 400
        
    tree_preds = np.array([est.predict(inp)[0] for est in model.estimators_])
    
    mean_pred = float(np.mean(tree_preds))
    median_pred = float(np.median(tree_preds))

    bin_width = int(data.get("bin_width", 10)) 
    
    min_score = max(0, int(tree_preds.min()) - bin_width)
    max_score = int(tree_preds.max()) + bin_width
    
    bins = list(range(min_score, max_score + bin_width, bin_width))
    counts, edges = np.histogram(tree_preds, bins=bins)
    
    labels = []
    probs = []
    
    total_trees = float(counts.sum()) if counts.sum() else 1.0
    for i in range(len(counts)):
        low = int(edges[i])
        high = int(edges[i+1]) - 1 
        labels.append(f"{low}-{high}")
        probs.append(float(counts[i]) / total_trees)

    response = {
        "mean": round(mean_pred, 1),
        "median": round(median_pred, 1),
        "bins": labels,
        "probs": [round(p, 4) for p in probs], 
        "range_low": max(0, round(mean_pred - np.std(tree_preds), 1)),
        "range_high": round(mean_pred + np.std(tree_preds), 1)
    }
    
    return jsonify(response)

@app.route('/static/<path:filename>')
def staticfiles(filename):
    return send_from_directory('static', filename)

if __name__ == "__main__":
    app.run(debug=True)