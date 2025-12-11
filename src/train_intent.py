# src/train_intent.py
import json, os, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

DATA_PATH = "data/intents.json"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data(path):
  with open(path, encoding="utf-8") as f:
    data = json.load(f)
  X, y = [], []
  for intent in data["intents"]:
    tag = intent["tag"]
    for ex in intent["examples"]:
      X.append(ex.lower())
      y.append(tag)
  return X, y

def train():
  X, y = load_data(DATA_PATH)
  pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2))),
                   ("clf", LogisticRegression(max_iter=2000))])
  pipe.fit(X,y)
  joblib.dump(pipe, os.path.join(MODEL_DIR,"intent_pipeline.joblib"))
  print("Saved model.")

if __name__ == "__main__":
  train()
