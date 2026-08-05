# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native configuration for DMTD Qwen3 checkpoints.

Two orthogonal axes select the variant:

- ``dmtd_block_attention``: how the ``tau`` slots of the current cycle see each
  other. ``"causal"``/``"bidirectional"`` are the Parallel variants, which run
  the parallel layers over a ``[head, MASK, ..., MASK]`` lookahead block.
  ``"none"`` is the original DMTD of arXiv:2510.11958: no lookahead block at
  all, so only the cycle head gets a parallel-layer representation and the
  other ``tau - 1`` positions enter the sequential layers on their own token
  embedding alone.
- ``dmtd_history_mode``: ``"real"`` recomputes the skipped positions once per
  cycle (the paper's cyclical refilling), ``"shadow"`` never does.
"""

from dataclasses import field

from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters


@strict
class DMTDQwen3Config(PreTrainedConfig):
    model_type = "dmtdqwen3"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 151936
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int | None = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 40960
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = True
    rope_parameters: RopeParameters | dict | None = field(
        default_factory=lambda: {"rope_type": "default", "rope_theta": 1000000}
    )
    attention_bias: bool = False
    use_sliding_window: bool = False
    sliding_window: int | None = 4096
    max_window_layers: int = 36
    layer_types: list[str] | None = None
    attention_dropout: float | int = 0.0
    pad_token_id: int | None = None
    bos_token_id: int | None = 151643
    eos_token_id: int | list[int] | None = 151645

    num_parallel_layers: int = 28
    num_sequential_layers: int = 8
    mtp_horizon: int = 4
    mask_token_id: int = 151660
    # ``None`` means "infer from the checkpoint's field naming" (see
    # `_resolve_layer_split`): the vanilla DMTD release describes its layer
    # split with the paper's encoding/thinking/decoding names and carries
    # neither knob, and it is the "none" + "real" variant.
    dmtd_history_mode: str | None = None
    dmtd_block_attention: str | None = None

    # The original DMTD release (xuan-luo/DMTD-Qwen3-4B) names the same split
    # after the paper's three layer groups. Accepted as aliases so that
    # checkpoint's own config.json loads unmodified.
    num_encoding_layers: int | None = None
    num_thinking_layers: int | None = None
    num_decoding_layers: int | None = None

    def _resolve_layer_split(self) -> bool:
        """Fold the paper's layer-group names into P28/S8. Returns whether the
        config used them, which is also what identifies a vanilla checkpoint."""
        if self.num_encoding_layers not in (None, 0):
            raise ValueError(
                "num_encoding_layers > 0 is not supported: this implementation "
                "has only parallel (thinking) and sequential (decoding) layer "
                f"groups; got {self.num_encoding_layers}."
            )
        vanilla_naming = (
            self.num_thinking_layers is not None or self.num_decoding_layers is not None
        )
        if self.num_thinking_layers is not None:
            self.num_parallel_layers = self.num_thinking_layers
        if self.num_decoding_layers is not None:
            self.num_sequential_layers = self.num_decoding_layers
        return vanilla_naming

    def __post_init__(self, **kwargs):
        vanilla_naming = self._resolve_layer_split()
        if self.dmtd_block_attention is None:
            self.dmtd_block_attention = "none" if vanilla_naming else "causal"
        if self.dmtd_history_mode is None:
            self.dmtd_history_mode = "real" if vanilla_naming else "shadow"
        self.sliding_window = self.sliding_window if self.use_sliding_window else None
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        if self.num_parallel_layers <= 0 or self.num_sequential_layers <= 0:
            raise ValueError(
                "num_parallel_layers and num_sequential_layers must both be "
                f"positive; got {self.num_parallel_layers} parallel / "
                f"{self.num_sequential_layers} sequential."
            )
        total_split = self.num_parallel_layers + self.num_sequential_layers
        if total_split != self.num_hidden_layers:
            raise ValueError(
                "num_parallel_layers + num_sequential_layers must equal "
                f"num_hidden_layers; got {self.num_parallel_layers} + "
                f"{self.num_sequential_layers} = {total_split} vs "
                f"num_hidden_layers={self.num_hidden_layers}."
            )
        if self.mtp_horizon < 1:
            raise ValueError(f"mtp_horizon must be >= 1, got {self.mtp_horizon}.")
        if not (0 <= self.mask_token_id < self.vocab_size):
            raise ValueError(
                "mask_token_id must be in [0, vocab_size); got "
                f"{self.mask_token_id} vs vocab_size={self.vocab_size}."
            )
        if self.dmtd_history_mode not in {"shadow", "real"}:
            raise ValueError(
                "dmtd_history_mode must be 'shadow' or 'real', got "
                f"{self.dmtd_history_mode!r}."
            )
        if self.dmtd_block_attention not in {"causal", "bidirectional", "none"}:
            raise ValueError(
                "dmtd_block_attention must be 'causal', 'bidirectional' or "
                f"'none', got {self.dmtd_block_attention!r}."
            )

        super().__post_init__(**kwargs)


__all__ = ["DMTDQwen3Config"]
