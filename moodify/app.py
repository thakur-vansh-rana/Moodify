from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import os

app = Flask(__name__, static_folder=".")
CORS(app)

# Load model files — place them in the same folder as this script
BASE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE, "logistic_regression_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE, "tfidf_vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

with open(os.path.join(BASE, "numeric_emotion_mapping.pkl"), "rb") as f:
    emotion_mapping = pickle.load(f)

# Reverse mapping: number -> emotion label
id_to_emotion = {v: k for k, v in emotion_mapping.items()}

EMOTION_META = {
    "joy":      {"emoji": "😄", "color": "#f59e0b"},
    "sadness":  {"emoji": "😢", "color": "#60a5fa"},
    "anger":    {"emoji": "😠", "color": "#f87171"},
    "fear":     {"emoji": "😨", "color": "#a78bfa"},
    "love":     {"emoji": "❤️",  "color": "#f472b6"},
    "surprise": {"emoji": "😲", "color": "#34d399"},
}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    X = vectorizer.transform([text])
    pred_id = int(model.predict(X)[0])
    probs = model.predict_proba(X)[0]

    emotion = id_to_emotion[pred_id]
    confidence = float(probs[pred_id])

    all_scores = [
        {
            "emotion": id_to_emotion[i],
            "score": float(p),
            "emoji": EMOTION_META[id_to_emotion[i]]["emoji"],
            "color": EMOTION_META[id_to_emotion[i]]["color"],
        }
        for i, p in enumerate(probs)
    ]
    all_scores.sort(key=lambda x: x["score"], reverse=True)

    return jsonify({
        "emotion": emotion,
        "emoji": EMOTION_META[emotion]["emoji"],
        "color": EMOTION_META[emotion]["color"],
        "confidence": confidence,
        "all_scores": all_scores,
    })

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

if __name__ == "__main__":
    print("🚀  Emotion Detector running at http://localhost:5000")
    app.run(debug=True, port=5000)
