"""Launch the gemma-4-31b-it judge as a vLLM OpenAI server with the heterogeneous-config fix.

`vllm serve` only takes a STATIC --hf-overrides dict, but gemma-4 on transformers 5.15 needs a
runtime patch: set allow_global_per_layer_attribute_access on the config AND its text_config, so
transformers' heterogeneity guard stops raising AmbiguousGlobalPerLayerAttributeError when vLLM
reads head_dim / num_key_value_heads. This is the exact callable used in discourse generation
(incident_sdf/discourse/run_generate.py:_hetero_override), passed here as a callable hf_overrides
into the OpenAI server's engine args. All other flags pass through like `vllm serve`.

  python scripts/serve_gemma_judge.py --model google/gemma-4-31b-it --served-model-name gemma \
      --port 28341 --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.90 \
      --enforce-eager --limit-mm-per-prompt '{"image":0,"audio":0}'
"""
from __future__ import annotations
import asyncio
import sys


def _hetero_override(config):
    """Verbatim behavior of run_generate._hetero_override: allow global-per-layer attribute access
    and restore global_head_dim / num_global_key_value_heads from the first full-attention layer."""
    cands = [config] + ([config.text_config] if hasattr(config, "text_config") else [])
    for c in cands:
        try:
            c.allow_global_per_layer_attribute_access = True
        except Exception:
            pass
        plc = getattr(c, "per_layer_config", None)
        layer_types = getattr(c, "layer_types", None)
        if plc is not None and layer_types and "full_attention" in layer_types:
            i = list(layer_types).index("full_attention")
            for name, src in (("global_head_dim", "head_dim"), ("num_global_key_value_heads", "num_key_value_heads")):
                try:
                    getattr(c, name)
                except AttributeError:
                    try:
                        setattr(c, name, getattr(plc[i], src))
                    except Exception:
                        pass
    return config


def main() -> None:
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm.entrypoints.openai.api_server import run_server
    try:                                                            # vLLM moved this across versions
        from vllm.entrypoints.openai.cli_args import FlexibleArgumentParser
    except ImportError:
        try:
            from vllm.utils.argparse_utils import FlexibleArgumentParser
        except ImportError:
            from vllm.utils import FlexibleArgumentParser

    parser = make_arg_parser(FlexibleArgumentParser(description="gemma judge server"))
    args = parser.parse_args(sys.argv[1:])
    # inject the callable override (CLI can only carry a static dict)
    args.hf_overrides = _hetero_override
    print(f"[serve_gemma_judge] model={args.model} port={args.port} enforce_eager={getattr(args,'enforce_eager',None)} "
          f"hf_overrides=<callable _hetero_override>", flush=True)
    asyncio.run(run_server(args))


if __name__ == "__main__":
    main()
