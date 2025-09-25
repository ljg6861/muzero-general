"""Lightweight entailment-based fact verifier."""

from __future__ import annotations

import os
from typing import Protocol
from contextlib import contextmanager

try:
    import torch
except ImportError:  # pragma: no cover - allow running without torch
    class _TorchStub:
        def no_grad(self):
            @contextmanager
            def _ctx():
                yield

            return _ctx()

    torch = _TorchStub()  # type: ignore[assignment]


class _EntailmentModel(Protocol):
    def predict_proba(self, premise: str, hypothesis: str):
        ...


class FactVerifier:
    """Wrapper around an NLI model for fact entailment checks."""

    def __init__(self, nli_model: _EntailmentModel) -> None:
        self.nli = nli_model
        self.threshold_loose = float(os.getenv("NLI_THRESH_LOOSE", "0.60"))
        self.threshold_strict = float(os.getenv("NLI_THRESH_STRICT", "0.80"))

    @torch.no_grad()
    def entailment_score(self, premise: str, hypothesis: str) -> float:
        """Return entailment probability in [0, 1]."""

        if not hasattr(self.nli, "predict_proba"):
            raise AttributeError("Underlying NLI model must expose predict_proba")
        probs = self.nli.predict_proba(premise, hypothesis)
        if probs is None or len(probs) == 0:
            return 0.0
        return float(probs[-1])


__all__ = ["FactVerifier"]
