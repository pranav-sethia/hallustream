"""Checks the two facts the whole project relies on.

1. ``output_hidden_states=True`` returns num_layers + 1 tensors: index 0 is the
   embeddings, index k (1 <= k < num_layers) is the output of decoder layer k-1,
   and the last entry is the final, post-norm state.
2. A manual decode loop with a KV cache returns hidden states for just the new
   token at each step, and they match a single full forward pass.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(TINY)
    return model, tok


def test_hidden_state_indexing(tiny):
    model, tok = tiny
    n_layers = model.config.num_hidden_layers
    ids = tok("The capital of France is Paris.", return_tensors="pt").input_ids

    captured = {}

    def hook(idx):
        def fn(_module, _inputs, output):
            captured[idx] = output[0] if isinstance(output, tuple) else output

        return fn

    handles = [layer.register_forward_hook(hook(i)) for i, layer in enumerate(model.model.layers)]
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    for h in handles:
        h.remove()

    hs = out.hidden_states
    assert len(hs) == n_layers + 1
    # hidden_states[k] == output of decoder layer k-1, for every layer except the last
    for k in range(1, n_layers):
        torch.testing.assert_close(hs[k], captured[k - 1])
    # the last entry is the post-norm final state, not the raw last-layer output
    torch.testing.assert_close(hs[-1], model.model.norm(captured[n_layers - 1]))


def test_manual_decode_matches_full_pass(tiny):
    model, tok = tiny
    layer = model.config.num_hidden_layers - 1  # any pre-norm index works
    ids = tok("Ada Lovelace was born in", return_tensors="pt").input_ids

    cache = DynamicCache()
    with torch.no_grad():
        out = model(ids, past_key_values=cache, use_cache=True)
        generated, step_states = [], []
        for _ in range(5):
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            generated.append(nxt)
            out = model(nxt, past_key_values=cache, use_cache=True, output_hidden_states=True)
            assert out.hidden_states[layer].shape[1] == 1  # only the new token
            step_states.append(out.hidden_states[layer][:, -1])

        full_ids = torch.cat([ids, *generated], dim=1)
        full = model(full_ids, output_hidden_states=True).hidden_states[layer]

    streamed = torch.stack(step_states, dim=1)
    expected = full[:, ids.shape[1] : ids.shape[1] + 5]
    torch.testing.assert_close(streamed, expected, atol=1e-4, rtol=1e-4)
