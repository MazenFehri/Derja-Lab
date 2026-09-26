from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression

from features.sentiment.preprocess import tokenize


class LogRegSentiment:
    """Each text -> [sum of positive word freqs, sum of negative word freqs] -> logistic regression."""

    name = "Logistic regression"

    def fit(self, texts, labels):
        self.pos, self.neg = Counter(), Counter()
        for text, label in zip(texts, labels):
            (self.pos if label == 1 else self.neg).update(tokenize(text))
        self.vocab = set(self.pos) | set(self.neg)
        self.clf = LogisticRegression(max_iter=1000).fit(self._features(texts), labels)
        return self

    def _features(self, texts):
        X = [[sum(self.pos[w] for w in tokenize(t)), sum(self.neg[w] for w in tokenize(t))] for t in texts]
        return np.log1p(np.array(X, dtype=float))  # long comments give huge raw counts; log keeps them comparable

    def predict_proba(self, text):
        """Probability that the text is positive."""
        return float(self.clf.predict_proba(self._features([text]))[0, 1])

    def score(self, texts, labels):
        return float(self.clf.score(self._features(texts), labels))
