"""Run Evo 2 WITHOUT the flash-attn package (e.g. on Colab, where
`pip install flash-attn` usually finds no prebuilt wheel for Colab's torch and
spends hours compiling).

Two facts from Vortex (vtx 1.0.8) make this work:
- With `use_flash_attn: False`, Vortex's MHA uses `SelfAttention`, which calls
  `torch.nn.functional.scaled_dot_product_attention`. On an A100 that runs
  PyTorch's own fused flash kernel, so memory stays linear in sequence length.
- The only import-time dependency is `vortex/ops/__init__.py` doing
  `import flash_attn_2_cuda` (pulled in by the rotary kernel's package). The
  module is only *called* inside flash code paths, so an empty placeholder
  module satisfies the import and is never touched with use_flash_attn=False.

`prepare()` must run before anything imports `evo2` or `vortex`.
"""
from __future__ import annotations

import importlib.util
import sys
import types


def flash_attn_available() -> bool:
    return importlib.util.find_spec("flash_attn_2_cuda") is not None


def ignore_broken_transformer_engine() -> None:
    """Colab's image ships an empty `transformer-engine` metapackage whose import
    raises RuntimeError. evo2 and vortex only guard `import transformer_engine`
    with `except ImportError`, so they crash. If the import fails with anything
    other than ImportError, make it an ImportError (sys.modules[name] = None);
    the 7B models run in bf16 without Transformer Engine anyway."""
    try:
        import transformer_engine  # noqa: F401
    except ImportError:
        return
    except Exception as e:  # noqa: BLE001
        for k in [k for k in sys.modules if k == "transformer_engine" or k.startswith("transformer_engine.")]:
            del sys.modules[k]
        sys.modules["transformer_engine"] = None
        print(f"marv_hyena: ignoring broken transformer_engine install ({type(e).__name__}); "
              "7B models run without it")


def prepare() -> None:
    """Placeholder flash_attn_2_cuda + force use_flash_attn=False when evo2 builds a model."""
    ignore_broken_transformer_engine()
    if "vortex" in sys.modules and "flash_attn_2_cuda" not in sys.modules:
        raise RuntimeError("vortex was imported before noflash.prepare(); restart the runtime and call prepare() first")
    if not flash_attn_available() and "flash_attn_2_cuda" not in sys.modules:
        stub = types.ModuleType("flash_attn_2_cuda")
        stub.__doc__ = "placeholder installed by marv_hyena.noflash; flash attention is disabled"
        sys.modules["flash_attn_2_cuda"] = stub

    import evo2.models as em

    if getattr(em.StripedHyena, "_marv_noflash", False):
        return
    original = em.StripedHyena

    def striped_hyena_without_flash(config, *args, **kwargs):
        config["use_flash_attn"] = False
        return original(config, *args, **kwargs)

    striped_hyena_without_flash._marv_noflash = True
    em.StripedHyena = striped_hyena_without_flash
