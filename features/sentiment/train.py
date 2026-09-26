import os
import pickle

import pandas as pd

from features.sentiment.logreg import LogRegSentiment

# add a model: write a class with fit / predict_proba / score / vocab / name, then register it here
MODELS = {
    "logreg": LogRegSentiment,
}

data = pd.read_json("data/preprocessed_data.jsonl", lines=True)
data = data[["src", "label"]].rename(columns={"src": "text"})
data["label"] = (data["label"] == "pos").astype(int)

train_df = data.sample(frac=0.8, random_state=42)
test_df = data.drop(train_df.index)

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    for key, Model in MODELS.items():
        model = Model().fit(train_df["text"].tolist(), train_df["label"].tolist())
        accuracy = model.score(test_df["text"].tolist(), test_df["label"].tolist())
        with open(f"models/sentiment_{key}.pkl", "wb") as f:
            pickle.dump({"model": model, "accuracy": accuracy}, f)
        print(f"{key:10s} test accuracy {accuracy:.4f}  -> models/sentiment_{key}.pkl")
