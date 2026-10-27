"""The probe as the hallucination detector inside SHARS (Li et al., ICML 2026).

SHARS checks each sentence it writes: it splits the sentence into (entity, claim) propositions,
asks a detector how uncertain each one is, and keeps, rewrites or redoes the sentence. This
estimator scores each proposition's entity with the probe from one forward pass, instead of
writing a question and sampling answers (docs/RESULTS.md, all 182 test people: 44% fewer
unsupported claims than SHARS's own detector at about 1/140 of the detection time).

SHARS is not on PyPI. Clone github.com/TreeLLi/hallucination-rejection-sampling, put it on
sys.path, then:

    import generator, uncertainty                     # SHARS
    from hallustream.integrations.shars import register
    register(uncertainty)                             # adds "probe" to SHARS's estimators
    gen = generator.UncertaintyGenerator(model, uncertainty_estimator="probe",
                                         uncertainty_threshold=0.5988, ...)

The threshold 0.5988 and the "native" context are the settings of the tested run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hallustream.probe import Probe, char_to_tokens, forward

BIO_PROMPT = "Tell me a bio of {name}."  # the probe's training-style prompt ("native" context)


def estimator_class(uncertainty, probe: Probe | None = None, weights: str | Path = "qwen3-4b",
                    system_prompt: str | None = None):  # fmt: skip
    """A SHARS UncertaintyEstimator subclass that scores propositions with the probe.

    uncertainty: SHARS's uncertainty module. probe: a loaded Probe (default: the bundled
    weights). system_prompt: SHARS's system prompt, used when context="shars".
    """
    from hallustream.detector import _weights_dir

    class ProbeEstimator(uncertainty.UncertaintyEstimator):
        """Uncertainty of a proposition = probe probability that its entity is made up.

        context "native": the probe reads the text as it was trained (user turn
        `Tell me a bio of X.`, no system prompt); "shars": SHARS's own messages.
        signal "v3", "v1", or "surprise" (max token surprise, the free logprob baseline).
        """

        def __init__(self, signal="v3", context="native", **kwargs):
            super().__init__(**kwargs)
            self.signal, self.context = signal, context
            self.n_props = self.n_unlocated = 0  # propositions whose entity is not in the text
            if probe is not None:
                self.det = probe
            else:
                d = _weights_dir(weights)
                self.det = Probe(sorted(d.glob("probe_seed*.safetensors")), d / "v3.json",
                                 device=str(self.model.model.device))  # fmt: skip

        def sentence_scores(self, user_query, text_sofar, sentence, query_entity, props):
            if self.context == "native":
                msgs = [{"role": "user", "content": BIO_PROMPT.format(name=query_entity)}]
            else:
                msgs = [{"role": "user", "content": user_query}]
                if system_prompt:
                    msgs.insert(0, {"role": "system", "content": system_prompt})
            text = text_sofar + sentence
            fw = forward(self.model.model, self.model.tokenizer, self.det, msgs, text,
                         enable_thinking=False)  # fmt: skip
            out = []
            for prop in props:
                lo = sentence.find(prop.entity)
                span = char_to_tokens(fw, len(text_sofar) + lo,
                                      len(text_sofar) + lo + len(prop.entity))  # fmt: skip
                if lo < 0 or span is None:
                    out.append(None)
                    continue
                v1, v3 = self.det.span_scores(fw, [span])
                a, b = span
                out.append({"v1": float(v1[0]), "v3": float(v3[0]),
                            "surprise": float(np.nanmax(fw.nll[a : b + 1])),
                            "entropy": float(np.nanmax(fw.entropy[a : b + 1]))})  # fmt: skip
            return out

        def get_sent_uncertainty(self, user_query, text_sofar, sentence, query_entity):
            props = self.get_sent_props(sentence, query_entity)
            scores = self.sentence_scores(user_query, text_sofar, sentence, query_entity, props)
            kept = []
            self.n_props += len(props)
            for prop, s in zip(props, scores, strict=True):
                if s is None:
                    self.n_unlocated += 1
                    continue
                x = s[self.signal]
                prop.uncertainty = x if self.signal == "surprise" else float(1 / (1 + np.exp(-x)))
                prop.qas = {}
                kept.append(prop)
            return kept

    return ProbeEstimator


def register(uncertainty, name: str = "probe", **kwargs):
    """Add the probe estimator to SHARS's registry, so generators can ask for it by name."""
    cls = estimator_class(uncertainty, **kwargs)
    uncertainty.ESTIMATORS[name] = cls
    return cls
