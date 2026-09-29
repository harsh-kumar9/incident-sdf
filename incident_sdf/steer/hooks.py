"""Residual-stream steering hooks.

Layer convention = HuggingFace ``hidden_states`` indexing: index 0 is the embedding output, index L (1..n)
is the output of decoder block L-1, i.e. the residual stream ENTERING block L (Betley et al.'s "block-input
residual"). Steering at index L therefore adds the vector to the output of ``layers[L-1]`` at every position.
"""
from __future__ import annotations
from contextlib import contextmanager
import torch
from torch import nn


def find_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """The decoder block list of a (possibly PEFT-wrapped) causal LM."""
    cands = [(n, m) for n, m in model.named_modules()
             if isinstance(m, nn.ModuleList) and n.endswith("layers") and len(m) > 0 and hasattr(m[0], "mlp")]
    if not cands:
        raise ValueError("no decoder layer list found")
    return cands[0][1]


class SteeringHook:
    """Adds ``vector`` (already scaled) to the residual stream at hidden_states index ``layer``."""

    def __init__(self, model: nn.Module, layer: int, vector: torch.Tensor):
        layers = find_decoder_layers(model)
        if not 1 <= layer <= len(layers):
            raise ValueError(f"layer index must be in 1..{len(layers)} (hidden_states convention), got {layer}")
        self.module = layers[layer - 1]
        self.vector = vector
        self._handle = None

    def _hook(self, module, args, output):
        v = self.vector
        if isinstance(output, tuple):
            h = output[0]
            return (h + v.to(device=h.device, dtype=h.dtype),) + tuple(output[1:])
        return output + v.to(device=output.device, dtype=output.dtype)

    def __enter__(self):
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove(); self._handle = None


@contextmanager
def steer(model: nn.Module, layer: int, direction: torch.Tensor, strength: float, typical_norm: float):
    """Betley scaling: strength 1.0 == one typical residual-stream row norm at that layer, along ``direction``."""
    if strength == 0:
        yield; return
    unit = direction / (direction.norm() + 1e-8)
    with SteeringHook(model, layer, unit * (strength * typical_norm)):
        yield
