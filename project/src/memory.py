
from collections import deque

import numpy as np

DIM_NAMES = ["interdisciplinary difficulty", "knowledge relevance", "response time", "error pattern", "learning gain"]
# Semantic rule base R (reconstruction of Appendix B, whose rule table is not recoverable from the provided text).
RULES = [
    ("error pattern", lambda v: v > 0.5, "recent consecutive failures on related concepts: reduce difficulty, review prerequisites"),
    ("response time", lambda v: v > 1.0, "unusually long response times: possible cognitive overload, prefer smaller steps"),
    ("learning gain", lambda v: v < 0.0, "negative recent gain: forgetting is outpacing practice, schedule a review"),
    ("interdisciplinary difficulty", lambda v: v > 0.5, "bridging concept with high difficulty: add a scaffolding item first"),
    ("knowledge relevance", lambda v: v < 0.0, "current item is weakly related to the weakest concept: re-target"),
]


class Memory:
    def __init__(self, window=20, bias_threshold=0.25, min_events=5):
        self.episodic = deque(maxlen=window)
        self.bias = {}                       # concept -> list of (p_pred - y)
        self.thr, self.min_events = bias_threshold, min_events

    def log_step(self, t, item, feats, gate, p, y):
        self.episodic.append(dict(t=t, item=int(item), feats=[float(x) for x in feats], gate=[float(x) for x in gate], p=float(p), y=float(y)))

    def log_deviation(self, concept, p, y):
        self.bias.setdefault(int(concept), []).append(float(p) - float(y))

    def semantic_labels(self, feats, gate):
        top = int(np.argmax(gate))
        notes = [msg for name, cond, msg in RULES if cond(feats[DIM_NAMES.index(name)])]
        return dict(dominant_dimension=DIM_NAMES[top], rule_notes=notes)

    def concept_bias(self, c):
        v = self.bias.get(int(c), [])
        return float(np.mean(v)) if len(v) >= self.min_events else 0.0

    def correction(self, concept, name):
        b = self.concept_bias(concept)
        if b > self.thr:
            return (f"The model's mastery estimate for '{name}' leans optimistic (bias {b:.3f} > {self.thr}); "
                    f"recommend supplementing foundational exercises before advancing.")
        return None


class ExperienceStore:
    """Vector DB of (profile, strategy summary, value). Retrieval score = beta*cos + (1-beta)*value  (Eq. 32 reconstruction);
    value update = EMA toward realised normalised gain G_p (Eq. 33 reconstruction)."""
    def __init__(self, beta=0.5, lr=0.3):
        self.vecs, self.texts, self.values = [], [], []
        self.beta, self.lr = beta, lr

    def add(self, vec, text, gain):
        self.vecs.append(np.asarray(vec, dtype=np.float32)); self.texts.append(text); self.values.append(float(gain))

    def retrieve(self, q, M=3):
        if not self.vecs:
            return []
        V = np.stack(self.vecs); q = np.asarray(q, dtype=np.float32)
        cos = V @ q / (np.linalg.norm(V, axis=1) * np.linalg.norm(q) + 1e-9)
        score = self.beta * cos + (1 - self.beta) * np.array(self.values)
        idx = np.argsort(-score)[:M]
        return [(int(i), self.texts[i], float(score[i])) for i in idx]

    def update_value(self, i, gain):
        self.values[i] = (1 - self.lr) * self.values[i] + self.lr * float(gain)
