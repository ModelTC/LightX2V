"""Audio-only Base10 block execution with reusable compile and CUDA Graph state."""

from dataclasses import replace
from functools import partial

from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
from lightx2v.models.networks.autoregressive_h3.transformer_infer import H3AutoregressiveTransformerInferMixin
from lightx2v.models.networks.minimax_h3.infer.offload.transformer_infer import MiniMaxH3OffloadTransformerInfer
from lightx2v.models.networks.minimax_h3.infer.transformer_infer import MiniMaxH3TransformerInfer


class TaoMateBase10TransformerInfer(H3AutoregressiveTransformerInferMixin, MiniMaxH3TransformerInfer):
    """Capture dense Base10 blocks after online AdaLN is prepared."""

    def _run_prepared_block(self, hidden, temb, indices, cos, sin, modulation, *, block_idx, block, pre):
        prepared = replace(pre, hidden_states=hidden, temb=temb, adaln_indices=indices, rotary_emb=(cos, sin))
        # Dense Base10 attention has no layer-specific state. A constant index
        # avoids one Dynamo specialization per physical offload buffer upload.
        self.block_idx = 0
        return BaseTransformerInfer.run_block(self, block_idx, block, hidden, prepared, modulation)

    def run_block(self, block_idx, block, hidden_states, pre):
        modulation = self._compute_adaln_table(block, pre)
        runner = partial(self._run_prepared_block, block_idx=block_idx, block=block, pre=pre)
        inputs = (hidden_states, pre.temb, pre.adaln_indices, *pre.rotary_emb, modulation)
        if self.cuda_graph_stack is None or pre.sequence_parallel_state is not None or not hidden_states.is_cuda:
            return runner(*inputs)
        weights = self.graph_weight_bindings(block, hidden_states.device)
        variant = (id(block), tuple(tuple(tensor.shape) for tensor in inputs))
        return self.run_captured_block(
            (id(block), self.use_compile),
            runner,
            inputs,
            bindings=weights,
            variant=variant,
        )


class TaoMateBase10OffloadTransformerInfer(TaoMateBase10TransformerInfer, MiniMaxH3OffloadTransformerInfer):
    pass
