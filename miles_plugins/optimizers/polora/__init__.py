"""PoLoRA: a preconditioned, orthogonalized optimizer for LoRA fine-tuning.

Vendored from https://github.com/nikhilgsh/polora (Apache-2.0). Megatron wiring
lives in ``megatron_adapter``, imported lazily by the training stack so this
package stays importable without Megatron.
"""

from .optimizer import Polora, collect_lora_pairs

__all__ = ["Polora", "collect_lora_pairs"]
