import math
from collections import Counter, defaultdict
from typing import List, Tuple
import re

class SimpleTFIDFIndex:
    def __init__(self):
        self.docs: List[str] = []
        self.term_freqs: List[Counter] = []
        self.df: Counter = Counter()
        self.N = 0
        self.token_re = re.compile(r"[A-Za-z][A-Za-z0-9_]+")

    def add(self, text: str):
        tokens = self._tokenize(text)
        tf = Counter(tokens)
        self.docs.append(text)
        self.term_freqs.append(tf)
        for t in tf.keys():
            self.df[t] += 1
        self.N += 1

    def _tokenize(self, s: str) -> List[str]:
        return [t.lower() for t in self.token_re.findall(s)]

    def build_from_iter(self, it, max_docs: int = 5000):
        for i, t in enumerate(it):
            if i >= max_docs:
                break
            self.add(t)

    def _tfidf(self, tf: Counter) -> dict:
        vec = {}
        for term, f in tf.items():
            idf = math.log((1 + self.N) / (1 + self.df[term])) + 1.0
            vec[term] = (1 + math.log(1 + f)) * idf
        return vec

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[float, str]]:
        qtf = Counter(self._tokenize(query))
        qvec = self._tfidf(qtf)
        scores = []
        for tf, doc in zip(self.term_freqs, self.docs):
            dvec = self._tfidf(tf)
            score = 0.0
            for term, w in qvec.items():
                score += w * dvec.get(term, 0.0)
            if score > 0:
                scores.append((score, doc))
        scores.sort(key=lambda x: -x[0])
        return scores[:k]
