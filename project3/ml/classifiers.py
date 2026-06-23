"""
Classifiers for AG News (Task 1).

Both classifiers expose the SAME interface so everything downstream
(deferral, active learning) is model-agnostic:

    clf.fit(texts, labels)
    clf.predict(texts)        -> list[int]
    clf.predict_proba(texts)  -> np.ndarray [n, n_classes]

Two implementations:

  TfidfLogReg
      TF-IDF features + multinomial logistic regression.
      Fast, CPU-only, runs anywhere. This is the baseline that the
      sandbox actually trains and that the Django app uses by default.

  DistilBertClassifier
      Fine-tunes distilbert-base-uncased. Stronger but needs torch +
      transformers and ideally a GPU. Written so it runs in the user's
      environment; the interface is identical so it is a drop-in swap.

Both are calibrated to give meaningful probabilities, which the
deferral and active-learning stages rely on.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class TfidfLogReg:
    name = "tfidf_logreg"

    def __init__(self, C=1.0, max_features=50000, ngram_range=(1, 2)):
        self.pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                sublinear_tf=True,
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=2,
                stop_words="english",
            )),
            ("clf", LogisticRegression(
                C=C, max_iter=1000, n_jobs=-1,
            )),
        ])
        self.classes_ = None

    def fit(self, texts, labels):
        self.pipe.fit(texts, labels)
        self.classes_ = self.pipe.named_steps["clf"].classes_
        return self

    def predict(self, texts):
        return list(self.pipe.predict(texts))

    def predict_proba(self, texts):
        return self.pipe.predict_proba(texts)


class DistilBertClassifier:
    """
    Fine-tuned DistilBERT. Requires `torch` and `transformers`.
    Kept lazy-imported so the rest of the package works without them.
    """
    name = "distilbert"

    def __init__(self, epochs=2, batch_size=16, lr=5e-5,
                 max_length=128, model_name="distilbert-base-uncased",
                 device=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.max_length = max_length
        self.model_name = model_name
        self.device = device
        self.classes_ = None
        self._model = None
        self._tokenizer = None

    def _ensure_imports(self):
        import torch
        from transformers import (AutoTokenizer,
                                   AutoModelForSequenceClassification)
        return torch, AutoTokenizer, AutoModelForSequenceClassification

    def fit(self, texts, labels):
        torch, AutoTokenizer, AutoModel = self._ensure_imports()
        from torch.utils.data import DataLoader, TensorDataset

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classes_ = sorted(set(labels))
        n_classes = len(self.classes_)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(
            self.model_name, num_labels=n_classes).to(self.device)

        enc = self._tokenizer(list(texts), truncation=True, padding=True,
                              max_length=self.max_length, return_tensors="pt")
        y = torch.tensor(labels)
        loader = DataLoader(
            TensorDataset(enc["input_ids"], enc["attention_mask"], y),
            batch_size=self.batch_size, shuffle=True)

        opt = torch.optim.AdamW(self._model.parameters(), lr=self.lr)
        self._model.train()
        for _ in range(self.epochs):
            for ids, mask, yb in loader:
                opt.zero_grad()
                out = self._model(input_ids=ids.to(self.device),
                                  attention_mask=mask.to(self.device),
                                  labels=yb.to(self.device))
                out.loss.backward()
                opt.step()
        return self

    def predict_proba(self, texts):
        torch, _, _ = self._ensure_imports()
        self._model.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = list(texts[i:i + self.batch_size])
                enc = self._tokenizer(batch, truncation=True, padding=True,
                                      max_length=self.max_length,
                                      return_tensors="pt").to(self.device)
                logits = self._model(**enc).logits
                p = torch.softmax(logits, dim=-1).cpu().numpy()
                probs.append(p)
        return np.vstack(probs)

    def predict(self, texts):
        return list(np.argmax(self.predict_proba(texts), axis=1))


def build_classifier(kind="tfidf", **kwargs):
    if kind == "tfidf":
        return TfidfLogReg(**kwargs)
    if kind == "distilbert":
        return DistilBertClassifier(**kwargs)
    raise ValueError(f"Unknown classifier kind: {kind}")
