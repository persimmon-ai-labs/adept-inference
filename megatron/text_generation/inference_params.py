from typing import Optional, Dict, Tuple, Any

import torch
from torch import Tensor


class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""

    def __init__(
        self,
        max_batch_size: int,
        max_sequence_len: int,
        lengths_per_sample: Optional[Tensor] = None,
        fused_ft_kernel: bool = False,
    ) -> None:
        # fused_ft_kernel: whether to use FasterTransformer fused single-query attention kernel.
        """Note that offsets are set to zero and we always set the
        flag to allocate memory. After the first call, make sure to
        set this flag to False."""
        self.max_sequence_len = max_sequence_len
        self.max_batch_size = max_batch_size
        self.sequence_len_offset = 0
        self.batch_size_offset = 0
        self.key_value_memory_dict: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        # This is used incase of an encoder-decoder model. The encoder output can be cached
        # and reused for multiple decoder forward passes.
        self.encoder_hidden_state: Dict[Any, Any] = {}
        self.return_encoder_hidden_state = False
        self.fused_ft_kernel = fused_ft_kernel
        self.lengths_per_sample: Tensor = lengths_per_sample
        # Raise import error at initialization time instead of the 1st generation time.
        if fused_ft_kernel:
            try:
                # pylint: disable=import-outside-toplevel,unused-import
                import ft_attention  # type: ignore
            except ImportError as exc:
                raise ImportError("Please install ft_attention from the FlashAttention repo.") from exc

    def reset(self) -> None:
        self.sequence_len_offset = 0
        self.batch_size_offset = 0

    def swap_key_value_dict(self, batch_idx: Any) -> None:
        "swap between batches"
        if len(self.key_value_memory_dict) == 0:
            raise ValueError("should not swap when dict in empty")

        for layer_number in self.key_value_memory_dict.keys():  # pylint: disable=consider-using-dict-items
            inference_key_memory, inference_value_memory = self.key_value_memory_dict[layer_number]
            assert len(batch_idx) == inference_key_memory.shape[1]  ## make sure batch size is the same
            new_inference_key_memory = inference_key_memory[:, batch_idx]
            new_inference_value_memory = inference_value_memory[:, batch_idx]
            self.key_value_memory_dict[layer_number] = (new_inference_key_memory, new_inference_value_memory)
