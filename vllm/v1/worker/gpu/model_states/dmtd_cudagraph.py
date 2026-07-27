# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA graphs for DMTD's parallel-layer groups.

A decode cycle head is a uniform batch: every participating request contributes
exactly ``rows_per_req`` parallel-layer rows (see
``DMTDQwen3ModelState.DECODE_ROWS_PER_REQ``). That is the same property DFlash
gets from freezing ``num_query_per_req`` at config time, and it is what lets a
group's forward be captured per participating-request bucket with the shared
``uniform_token_count`` dispatch dimension -- no new descriptor field needed.

Each group gets its own manager because ``uniform_token_count`` differs per group
(and per variant), and because a group's ``causal`` flag is frozen into its
metadata at build time.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)

logger = init_logger(__name__)


class DMTDParallelCudaGraphManager(CudaGraphManager):
    """Full CUDA graphs for one DMTD parallel-layer group.

    Only the decode shape is captured, so the mode is forced to
    ``FULL_DECODE_ONLY``; prefill groups have a per-step row count and always run
    eager.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        *,
        kind: str,
        rows_per_req: int,
    ) -> None:
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        else:
            cudagraph_mode = CUDAGraphMode.NONE
        super().__init__(
            vllm_config,
            device,
            cudagraph_mode,
            decode_query_len=rows_per_req,
        )
        self.kind = kind
        self.rows_per_req = rows_per_req
        self.num_replays = 0
        self._desc_by_num_reqs: dict[int, BatchExecutionDescriptor] = {}
        # Bucket on the *participating request* count rather than reusing the
        # shared token-count ladder. That ladder is sized for the main model's
        # one-token-per-request decode, so `max_cudagraph_capture_size` would cut
        # this group off at `max_cudagraph_capture_size / rows_per_req` requests.
        self._captured_num_reqs = self._build_request_ladder()
        self._candidates = {}
        self._capture_descs = {}
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
            self._capture_descs = {
                CUDAGraphMode.FULL: [
                    BatchExecutionDescriptor(
                        cg_mode=CUDAGraphMode.FULL,
                        num_tokens=num_reqs * rows_per_req,
                        num_reqs=num_reqs,
                        uniform_token_count=rows_per_req,
                    )
                    for num_reqs in self._captured_num_reqs
                ]
            }

    def _build_request_ladder(self) -> list[int]:
        """Participating-request counts to capture, ascending.

        Powers of two up to `max_num_reqs`, which keeps the graph count
        logarithmic in the batch size while never padding a step by more than
        2x. At most one cycle head per request can be pending, so `max_num_reqs`
        is a hard upper bound.
        """
        ladder = {1, self.max_num_reqs}
        size = 2
        while size < self.max_num_reqs:
            ladder.add(size)
            size *= 2
        return sorted(n for n in ladder if n >= 1)

    def capture_num_reqs(self) -> list[int]:
        """Request counts graphs will be captured for."""
        return list(self._captured_num_reqs)

    def padded_num_reqs(self, num_reqs: int) -> int | None:
        """Smallest captured request count that can serve `num_reqs`, if any."""
        if not self._graphs_captured:
            return None
        for captured in self._captured_num_reqs:
            if captured >= num_reqs:
                return captured
        return None

    def can_replay(self, num_reqs: int) -> bool:
        return (
            self._graphs_captured
            and self._desc_by_num_reqs.get(num_reqs) in self.graphs
        )

    def replay(self, num_reqs: int) -> None:
        desc = self._desc_by_num_reqs[num_reqs]
        self.num_replays += 1
        super().run_fullgraph(desc)

    def capture(
        self,
        prepare_fn: Callable[[int], Callable[[], None]],
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        """Capture one graph per participating-request bucket.

        `prepare_fn(num_reqs)` sets up the group's persistent buffers for that
        bucket -- outside the graph, as the shared capture protocol requires --
        and returns the callable to record.
        """

        def create_forward_fn(desc: BatchExecutionDescriptor, warmup: bool):
            assert desc.num_reqs is not None
            self._desc_by_num_reqs[desc.num_reqs] = desc
            forward = prepare_fn(desc.num_reqs)
            return (lambda cg_mode: forward()), None

        super().capture(create_forward_fn, progress_bar_desc)
        logger.info(
            "DMTD captured %d %s parallel-layer graphs (%d rows per request)",
            len(self.graphs),
            self.kind,
            self.rows_per_req,
        )
