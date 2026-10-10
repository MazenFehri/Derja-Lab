from collections import Counter

import numpy as np

from features.sentiment.preprocess import tokenize


class NaiveBayesSentiment:
    """score = log prior + sum of word log-likelihoods (Laplace smoothing); positive if score > 0."""

    name = "Naive Bayes"

    def fit(self, texts, labels):
        self.pos, self.neg = Counter(), Counter()
        for text, label in zip(texts, labels):
            (self.pos if label == 1 else self.neg).update(tokenize(text))

        self.vocab = set(self.pos) | set(self.neg)
        labels = np.array(labels)
        p_pos = np.sum(labels == 1) / len(labels)
        p_neg = 1 - p_pos
        self.log_prior = np.log(p_pos / p_neg)

        n_pos = sum(self.pos.values())  # total word occurrences in positive comments
        n_neg = sum(self.neg.values())
        V = len(self.vocab)
        self.loglikelihood = {
            w: np.log((self.pos[w] + 1) / (n_pos + V)) - np.log((self.neg[w] + 1) / (n_neg + V))
            for w in self.vocab
        }
        return self

    def _score(self, text):
        """Log-odds of positive vs negative for one text. Unknown words add nothing."""
        return self.log_prior + sum(self.loglikelihood.get(w, 0) for w in tokenize(text))

    def predict_proba(self, text):
        """Probability that the text is positive: sigmoid of the log-odds."""
        return float(1 / (1 + np.exp(-np.clip(self._score(text), -30, 30))))  # clip: no overflow on long texts

    def score(self, texts, labels):
        preds = np.array([self._score(t) > 0 for t in texts])
        return float(np.mean(preds == np.array(labels)))
