# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native DMTD Qwen3 inference."""

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.v1.attention.backend import AttentionType
from vllm.v1.worker.gpu.model_states.dmtd_qwen3 import DMTDQwen3ModelState

from .interfaces import LocalArgmaxMixin
from .qwen3 import Qwen3Attention, Qwen3DecoderLayer, Qwen3MLP, Qwen3Model
from .utils import (
    AutoWeightsLoader,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)


class DMTDQwen3Attention(Qwen3Attention):
    """Qwen3 attention with numerically independent Q/K/V projections."""

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        weight = self.qkv_proj.weight
        bias = self.qkv_proj.bias
        q_end = self.q_size
        k_end = q_end + self.kv_size

        q = F.linear(
            hidden_states,
            weight[:q_end],
            None if bias is None else bias[:q_end],
        )
        k = F.linear(
            hidden_states,
            weight[q_end:k_end],
            None if bias is None else bias[q_end:k_end],
        )
        v = F.linear(
            hidden_states,
            weight[k_end:],
            None if bias is None else bias[k_end:],
        )

        q_by_head = q.view(*q.shape[:-1], -1, self.head_dim)
        q = self.q_norm(q_by_head).view(q.shape)
        k_by_head = k.view(*k.shape[:-1], -1, self.head_dim)
        k = self.k_norm(k_by_head).view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class DMTDQwen3DecoderLayer(Qwen3DecoderLayer):
    def __init__(
        self,
        config,
        cache_config=None,
        quant_config=None,
        prefix: str = "",
        per_layer_sliding_window: int | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        set_default_rope_theta(config, default_theta=1000000)
        self.self_attn = DMTDQwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
            attn_type=AttentionType.DECODER,
            dual_chunk_attention_config=getattr(
                config, "dual_chunk_attention_config", None
            ),
            per_layer_sliding_window=per_layer_sliding_window,
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


class DMTDQwen3Model(nn.Module):
    """Qwen3 with an explicit parallel-layer shadow/refresh branch and a sequential-layer real branch."""

    hf_to_vllm_mapper = Qwen3Model.hf_to_vllm_mapper

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config.get_text_config()
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config
        self.num_parallel_layers = config.num_parallel_layers
        self.num_sequential_layers = config.num_sequential_layers
        self.mtp_horizon = config.mtp_horizon
        bidirectional = (
            getattr(config, "dmtd_block_attention", "causal") == "bidirectional"
        )

        def make_decoder_layer(prefix: str) -> Qwen3DecoderLayer:
            # The bidirectional checkpoints are sensitive enough that a packed
            # BF16 QKV GEMM can alter later greedy tokens after cached
            # sequential-layer decoding. Match their three independent
            # Transformers projections.
            layer_cls = DMTDQwen3DecoderLayer if bidirectional else Qwen3DecoderLayer
            return layer_cls(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            )

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            make_decoder_layer,
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def _run_layers(
        self,
        start: int,
        end: int,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = None
        for layer in self.layers[start:end]:
            hidden_states, residual = layer(positions, hidden_states, residual)
        assert residual is not None
        return hidden_states, residual

    def _set_parallel_context(
        self,
        *,
        slot_mapping: torch.Tensor | None,
        attn_metadata: dict[str, Any] | None = None,
    ) -> None:
        context = get_forward_context()
        for layer in self.layers[: self.num_parallel_layers]:
            layer_name = layer.self_attn.attn.layer_name
            if attn_metadata is not None:
                context.attn_metadata[layer_name] = attn_metadata[layer_name]
            if slot_mapping is not None:
                context.slot_mapping[layer_name] = slot_mapping

    def _run_parallel_layers(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        *,
        slot_mapping: torch.Tensor | None,
        attn_metadata: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        self._set_parallel_context(slot_mapping=slot_mapping, attn_metadata=attn_metadata)
        hidden = self.embed_input_ids(input_ids)
        hidden, residual = self._run_layers(
            0,
            self.num_parallel_layers,
            positions,
            hidden,
        )
        return hidden + residual

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        *,
        dmtd_shadow_input_ids: torch.Tensor,
        dmtd_shadow_positions: torch.Tensor,
        dmtd_shadow_slot_mapping: torch.Tensor | None,
        dmtd_direct_indices: torch.Tensor,
        dmtd_seq_req_slots: torch.Tensor,
        dmtd_seq_phases: torch.Tensor,
        dmtd_cycle_hidden: torch.Tensor,
        dmtd_write_req_slots: torch.Tensor,
        dmtd_write_phases: torch.Tensor,
        dmtd_write_shadow_indices: torch.Tensor,
        dmtd_refresh_input_ids: torch.Tensor | None = None,
        dmtd_refresh_positions: torch.Tensor | None = None,
        dmtd_refresh_slot_mapping: torch.Tensor | None = None,
        dmtd_refresh_attn_metadata: dict[str, Any] | None = None,
        dmtd_parallel_segments: list[dict[str, Any]] | None = None,
        dmtd_refresh_segments: list[dict[str, Any]] | None = None,
        dmtd_history_mode: str = "shadow",
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            raise ValueError("DMTDQwen3 does not support pipeline parallelism.")
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embed_input_ids(input_ids)

        parallel_hidden = None
        if dmtd_shadow_input_ids.numel() > 0:
            history_mode = dmtd_history_mode or "shadow"
            two_pass = (
                history_mode == "real"
                and dmtd_refresh_input_ids is not None
                and dmtd_refresh_input_ids.numel() > 0
            )
            if two_pass:
                assert dmtd_refresh_positions is not None
                if dmtd_refresh_segments:
                    for seg in dmtd_refresh_segments:
                        self._run_parallel_layers(
                            dmtd_refresh_input_ids[seg["start"] : seg["end"]],
                            dmtd_refresh_positions[seg["start"] : seg["end"]],
                            slot_mapping=seg["slot_mapping"],
                            attn_metadata=seg["attn_metadata"],
                        )
                else:
                    self._run_parallel_layers(
                        dmtd_refresh_input_ids,
                        dmtd_refresh_positions,
                        slot_mapping=dmtd_refresh_slot_mapping,
                        attn_metadata=dmtd_refresh_attn_metadata,
                    )

            if dmtd_parallel_segments:
                pieces: list[torch.Tensor] = []
                for seg in dmtd_parallel_segments:
                    pieces.append(
                        self._run_parallel_layers(
                            dmtd_shadow_input_ids[seg["start"] : seg["end"]],
                            dmtd_shadow_positions[seg["start"] : seg["end"]],
                            slot_mapping=seg["slot_mapping"],
                            attn_metadata=seg["attn_metadata"],
                        )
                    )
                parallel_hidden = torch.cat(pieces, dim=0)
            else:
                parallel_hidden = self._run_parallel_layers(
                    dmtd_shadow_input_ids,
                    dmtd_shadow_positions,
                    slot_mapping=dmtd_shadow_slot_mapping,
                )

            # Only shadow positions are written into the cycle buffer; refresh
            # hiddens never enter the sequential layers.
            if dmtd_write_shadow_indices.numel() > 0:
                dmtd_cycle_hidden[dmtd_write_req_slots, dmtd_write_phases] = (
                    parallel_hidden[dmtd_write_shadow_indices]
                )

        sequential_shadow = torch.empty_like(inputs_embeds)
        direct = dmtd_direct_indices >= 0
        if direct.any():
            assert parallel_hidden is not None
            sequential_shadow[direct] = parallel_hidden[dmtd_direct_indices[direct]]
        buffered = ~direct
        if buffered.any():
            sequential_shadow[buffered] = dmtd_cycle_hidden[
                dmtd_seq_req_slots[buffered],
                dmtd_seq_phases[buffered],
            ]

        hidden_states = inputs_embeds + sequential_shadow
        hidden_states, residual = self._run_layers(
            self.num_parallel_layers,
            self.num_parallel_layers + self.num_sequential_layers,
            positions,
            hidden_states,
        )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class DMTDQwen3ForCausalLM(LocalArgmaxMixin, nn.Module):
    hf_to_vllm_mapper = DMTDQwen3Model.hf_to_vllm_mapper
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        if not vllm_config.use_v2_model_runner:
            raise ValueError("DMTDQwen3ForCausalLM requires the V2 model runner.")

        config = vllm_config.model_config.hf_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.model = DMTDQwen3Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    @staticmethod
    def get_model_state_cls() -> type[DMTDQwen3ModelState]:
        return DMTDQwen3ModelState

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **kwargs,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
