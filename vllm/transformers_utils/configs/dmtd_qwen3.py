# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native configuration for Parallel DMTD Qwen3 checkpoints."""

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
    dmtd_history_mode: str = "shadow"
    dmtd_block_attention: str = "causal"

    def __post_init__(self, **kwargs):
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
        if self.dmtd_block_attention not in {"causal", "bidirectional"}:
            raise ValueError(
                "dmtd_block_attention must be 'causal' or 'bidirectional', got "
                f"{self.dmtd_block_attention!r}."
            )

        super().__post_init__(**kwargs)


__all__ = ["DMTDQwen3Config"]
