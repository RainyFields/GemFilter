# transformers.__version__ == '4.43.3'
import math
import typing
import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, TypedDict
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3RMSNorm
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb,eager_attention_forward

from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from .gem_filter_utils import find_context


import sys
if sys.version_info >= (3, 11):
    Unpack = typing.Unpack
else:
    Unpack = typing_extensions.Unpack


class FlashAttentionKwargs(TypedDict, total=False):
    """
    Keyword arguments for Flash Attention with Compile.

    Attributes:
        cumulative_seqlens_q (`torch.LongTensor`, *optional*)
            Gets cumulative sequence length for query state.
        cumulative_seqlens_k (`torch.LongTensor`, *optional*)
            Gets cumulative sequence length for key state.
        max_length_q (`int`, *optional*):
            Maximum sequence length for query state.
        max_length_k (`int`, *optional*):
            Maximum sequence length for key state.
    """

    cumulative_seqlens_q: Optional[torch.LongTensor]
    cumulative_seqlens_k: Optional[torch.LongTensor]
    max_length_q: Optional[int]
    max_length_k: Optional[int]


class Qwen3SelectAttention(Qwen3Attention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
        self.reset()
        self.topk = 1024
        self.select_layer_idx = 13
        self.select_mode = False

    def reset(self):
        self.indecies = None  # Make sure 'indecies' is set to None initially
        return

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # [GEMFILTER] Only populate 'indecies' in select_mode
        if self.select_mode:
            self.reset()  # Reset the indecies each time
            # print("during 1st forward pass, select_mode is True")
            find_context(self, query_states, key_states)
            # print(f"during 1st forward pass, indecies: {self.indecies}")
            # Here, we assume that 'find_context' returns or modifies 'self.indecies'
            # if self.indecies is None:
                # print("Warning: indecies are still None in select_mode")

        if not self.select_mode and past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        # [GEMFILTER] End

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # different from Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
