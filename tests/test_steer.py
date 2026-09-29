"""Steering machinery on a tiny random Qwen3 (CPU). Needs transformers>=5, peft; the Qwen3-4B tokenizer must be
cached (ada sote env; HF_HUB_OFFLINE=1). Run on ada: python -m pytest tests/test_steer.py -q"""
from __future__ import annotations
import pytest

pytest.importorskip("peft")
tf = pytest.importorskip("transformers")
import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from incident_sdf.steer import axes as A  # noqa: E402
from incident_sdf.steer import extract as X  # noqa: E402
from incident_sdf.steer import measures as M  # noqa: E402
from incident_sdf.steer.hooks import SteeringHook, find_decoder_layers, steer  # noqa: E402


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"tokenizer not cached: {e}")


@pytest.fixture
def tiny(tok):
    cfg = Qwen3Config(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, vocab_size=len(tok), max_position_embeddings=512,
                      tie_word_embeddings=False)
    torch.manual_seed(0)
    return Qwen3ForCausalLM(cfg).float().eval()


def test_axes_are_well_formed():
    for ax, d in A.AXES.items():
        if ax == "random":
            continue
        ps = A.pairs(ax)
        assert len(ps) == 260 and d["cls"] in {"incident", "generic", "auditor", "control"}
        for p, n, i in ps:
            (tp, sp), (tn, sn) = p.rsplit("\n\n", 1), n.rsplit("\n\n", 1)
            assert tp == tn and sp != sn and sp == d["pos"][i] and sn == d["neg"][i]
        q, pos, neg = d["concept"]
        assert q.endswith("?") and pos != neg
        assert len(A.concept_prompts(ax)) == 2
    assert not set(A.HELDOUT_TASKS) & set(A.EXTRACT_TASKS)


def test_find_layers_plain_and_peft(tiny):
    assert len(find_decoder_layers(tiny)) == 4
    p = get_peft_model(tiny, LoraConfig(r=2, target_modules=["q_proj"], task_type="CAUSAL_LM"))
    assert len(find_decoder_layers(p)) == 4


def test_hook_identity_at_zero_changes_otherwise_and_is_removed(tiny, tok):
    enc = tok(["hello world"], return_tensors="pt")
    with torch.no_grad():
        base = tiny(**enc).logits
        with steer(tiny, 2, torch.ones(64), 0.0, 10.0):
            assert torch.equal(tiny(**enc).logits, base)
        with steer(tiny, 2, torch.ones(64), 1.0, 10.0):
            assert not torch.allclose(tiny(**enc).logits, base)
        assert torch.equal(tiny(**enc).logits, base)
    assert find_decoder_layers(tiny)[1]._forward_hooks == {}
    with pytest.raises(ValueError):
        SteeringHook(tiny, 0, torch.ones(64))


def test_steer_scaling_uses_unit_direction_times_scale(tiny, tok):
    enc = tok(["abc"], return_tensors="pt")
    captured = {}
    layer = find_decoder_layers(tiny)[1]
    h = layer.register_forward_hook(lambda m, i, o: captured.setdefault("plain", (o[0] if isinstance(o, tuple) else o).clone()))
    with torch.no_grad():
        tiny(**enc)
    h.remove()
    with steer(tiny, 2, torch.tensor([3.0] + [0.0] * 63), 0.5, 8.0):
        h = layer.register_forward_hook(lambda m, i, o: captured.setdefault("steered", (o[0] if isinstance(o, tuple) else o).clone()))
        with torch.no_grad():
            tiny(**enc)
        h.remove()
    delta = captured["steered"] - captured["plain"]
    assert torch.allclose(delta[..., 0], torch.full_like(delta[..., 0], 4.0), atol=1e-5)
    assert torch.allclose(delta[..., 1:], torch.zeros_like(delta[..., 1:]), atol=1e-5)


def test_extract_shapes_antisymmetry_and_neutral_bank(tiny, tok):
    d = X.extract_axis(tiny, tok, "grader", batch=4, n_pairs=10)
    assert d["vector"].shape == (5, 64) and d["half_a"].shape == (5, 64) and d["n_pairs"] == 10
    assert (d["typical_norm"] > 0).all() and 0 <= float(d["consistency"][2]) <= 1
    A.AXES["_flip"] = {**A.AXES["grader"], "pos": A.AXES["grader"]["neg"], "neg": A.AXES["grader"]["pos"]}
    try:
        f = X.extract_axis(tiny, tok, "_flip", batch=4, n_pairs=10)
    finally:
        del A.AXES["_flip"]
    assert torch.allclose(f["vector"], -d["vector"], atol=1e-4)
    ns = X.neutral_stats(tiny, tok, batch=8, bank_size=12, readout_prompts=["Pick A or B."])
    assert ns["mean"].shape == (5, 64) and ns["n"] == 20 and ns["bank"].shape == (12, 5, 64) and "readout_mean" in ns
    r = X.covariance_random(ns["bank"], 2, seed=1)
    assert r.shape == (64,) and torch.isfinite(r).all() and not torch.equal(r, X.covariance_random(ns["bank"], 2, seed=2))


def test_batched_two_choice_matches_single_prompt_and_is_position_bias_free(tiny, tok):
    items = [{"question": f"Q{i}: pick one.", "choices": ["do the honest thing", "do the sneaky thing"], "misaligned_choice": 1}
             for i in range(5)]
    items[2]["question"] = "A much longer question " * 8 + "pick one."   # forces left padding
    lids = M.letter_ids(tok)
    assert all(len(v) >= 1 for v in lids.values())
    r = M.propensity(tiny, tok, items, batch=4)
    assert len(r["items"]) == 5 == len(r["items_d"]) and 0 <= r["p_mis"] <= 1 and 0 <= r["validity"] <= 1
    pp = M.propensity_prompts(items)
    vals = []
    for prompt, mis in pp[4:6]:                       # the padded item, both orderings, single-prompt reference
        enc = tok(M.render(tok, prompt), return_tensors="pt")
        with torch.no_grad():
            lg = tiny(**enc).logits[0, -1]
        full = torch.softmax(lg, 0)
        pa, pb = full[lids["A"]].sum(), full[lids["B"]].sum()
        p_yes, p_no = (pa, pb) if mis == 0 else (pb, pa)
        vals.append((float(p_yes / (p_yes + p_no)), float(torch.log(p_yes) - torch.log(p_no))))
    assert abs(r["items"][2] - (vals[0][0] + vals[1][0]) / 2) < 1e-3
    assert abs(r["items_d"][2] - (vals[0][1] + vals[1][1]) / 2) < 1e-3


def test_concept_and_darktriad_shapes(tiny, tok):
    c = M.concept(tiny, tok, ["grader", "locale"], batch=4)
    assert set(c) == {"grader", "locale"} and 0 <= c["grader"]["p_pos"] <= 1
    data = {t: [{"question": "Scenario.", "response_high1": "h1", "response_high2": "h2", "response_low1": "l1", "response_low2": "l2"}]
            for t in M.TRAITS}
    r = M.darktriad(tiny, tok, data, batch=4)
    assert set(M.TRAITS) <= set(r) and 0 <= r["darktriad_mean"] <= 1 and len(r["Narcissism"]["items_d"]) == 1


def test_subset_is_deterministic():
    rows = list(range(100))
    assert M.subset(rows, 10) == M.subset(rows, 10) and len(M.subset(rows, 10)) == 10 and M.subset(rows, None) == rows
