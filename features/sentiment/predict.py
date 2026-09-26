import pickle
from pathlib import Path

from features.sentiment.preprocess import tokenize

# every trained model in models/ shows up on the site automatically
MODELS = {}
for path in sorted(Path("models").glob("sentiment_*.pkl")):
    with open(path, "rb") as f:
        MODELS[path.stem.removeprefix("sentiment_")] = pickle.load(f)


def available():
    return [{"id": key, "name": m["model"].name, "accuracy": m["accuracy"]} for key, m in MODELS.items()]


def predict(text, model_id):
    model = MODELS[model_id]["model"]
    words = tokenize(text)
    p = model.predict_proba(text)
    return {
        "label": "positive" if p >= 0.5 else "negative",
        "confidence": max(p, 1 - p),
        "known_words": sum(w in model.vocab for w in words),
        "total_words": len(words),
    }
