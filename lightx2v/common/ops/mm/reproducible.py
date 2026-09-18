import torch
import torch.distributed as dist

from lightx2v.common.ops.mm.mm_weight import MMWeightTP


class MMWeightTPReproducible(MMWeightTP):
    """Use the same eight logical GEMM partitions for H3 TP1/2/4/8."""

    def apply(self, input_tensor):
        weight = self._mm._get_actual_weight()
        if self.split_dim == "row":
            return self._apply_row(input_tensor, weight)
        return self._apply_col(input_tensor, weight, self._mm._get_actual_bias())

    def _apply_row(self, x, weight):
        chunk_k = x.shape[1] // (8 // self.tp_size)
        total = None
        for start in range(0, x.shape[1], chunk_k):
            a = x[:, start : start + chunk_k].contiguous()
            b = weight[start : start + chunk_k].T.contiguous().T
            partial = torch.mm(a, b, out_dtype=torch.float32)
            if total is None:
                total = partial.double()
            else:
                total.add_(partial)
        # FP64 limits rounding from different TP summation groupings.
        if self.tp_size > 1:
            dist.all_reduce(total, op=dist.ReduceOp.SUM, group=self.tp_group)
        if self._row_split_bias is not None:
            total.add_(self._row_split_bias)
        return total.to(x.dtype)

    def _apply_col(self, x, weight, bias):
        local_parts = 8 // self.tp_size
        segments = self.lora_column_chunks
        segment_width = weight.shape[1] // segments
        part_width = segment_width // local_parts
        output = torch.empty((x.shape[0], weight.shape[1]), dtype=x.dtype, device=x.device)
        x = x.contiguous()
        for part in range(local_parts):
            slices = [slice(segment * segment_width + part * part_width, segment * segment_width + (part + 1) * part_width) for segment in range(segments)]
            # Keep each FFN GEMM in [value_rank, gate_rank] order.
            b = torch.cat([weight[:, index] for index in slices], dim=1).T.contiguous().T
            if bias is None:
                result = torch.mm(x, b)
            else:
                part_bias = torch.cat([bias[index] for index in slices]).contiguous()
                result = torch.addmm(part_bias, x, b)
            for segment, index in enumerate(slices):
                output[:, index] = result[:, segment * part_width : (segment + 1) * part_width]
        return output
