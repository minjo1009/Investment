
import numpy as np, json, bisect


class QuantileMap:
    def __init__(self, quantiles=None, values=None):
        self.q = quantiles or []
        self.v = values or []

    @staticmethod
    def fit(scores, n_bins=100):
        s = np.asarray(scores, float)
        qs = np.linspace(0, 1, n_bins + 1)
        vs = np.quantile(s, qs, method="linear")
        return QuantileMap(list(qs), list(vs))

    def transform(self, scores):
        s = np.asarray(scores, float)
        out = np.empty_like(s)
        for i, x in enumerate(s):
            j = bisect.bisect_left(self.v, x)
            if j <= 0:
                out[i] = self.q[0]
            elif j >= len(self.v):
                out[i] = self.q[-1]
            else:
                x0, x1 = self.v[j - 1], self.v[j]
                q0, q1 = self.q[j - 1], self.q[j]
                t = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
                out[i] = q0 + (q1 - q0) * t
        return out

    def dumps(self):
        return json.dumps({"q": self.q, "v": self.v})

    @staticmethod
    def loads(s):
        o = json.loads(s)
        return QuantileMap(o["q"], o["v"])
